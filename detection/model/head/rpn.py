import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import clip_boxes_to_image, batched_nms

from gvcore.model import GenericModule

from model.anchor.generator import AnchorGenerator
from model.anchor import BoxCoder, Matcher
from model.utils import match_label, sample_pos_neg

from utils.box import nonempty


class StandardRPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        """
        NOTE: this interface is experimental.

        Args:
            in_channels (int): number of input feature channels. When using multiple
                input features, they must have the same number of channels.
            num_anchors (int): number of anchors to predict for *each spatial position*
                on the feature map. The total number of anchors for each
                feature map will be `num_anchors * H * W`.
        """
        super().__init__()
        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        # torch.manual_seed(1)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(1)
        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, feats):
        objectness_logits, anchor_deltas = [], []
        for feat in feats:
            feat = F.relu(self.conv(feat))
            objectness_logits.append(self.objectness_logits(feat))
            anchor_deltas.append(self.anchor_deltas(feat))
        return objectness_logits, anchor_deltas


class RPN(GenericModule):
    def __init__(self, cfg):
        super(RPN, self).__init__(cfg)

        self.rpn_head = StandardRPNHead(256, 3)

        # Params ==============
        self.pre_nms_topk = (cfg.model.rpn.pre_nms_topk_train, cfg.model.rpn.pre_nms_topk_test)
        self.post_nms_topk = (cfg.model.rpn.post_nms_topk_train, cfg.model.rpn.post_nms_topk_test)
        self.nms_threshold = cfg.model.rpn.nms_threshold
        self.in_feats = ("p2", "p3", "p4", "p5", "p6")
        self.rpn_num = 256
        self.rpn_positive_fraction = 0.5

        anchor_sizes = tuple((x,) for x in [32, 64, 128, 256, 512])
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

        # Tools ===============
        self.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios, strides=(4, 8, 16, 32, 64), offset=0.0)
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.matcher = Matcher(thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True)

        # Buffer ==============
        self.anchors = None
        self.objectness_logits = None
        self.anchor_deltas = None
        self.proposals = None

    def _forward(self, feats, img_sizes):
        feats = [feats[f] for f in self.in_feats]

        self.anchors = self.anchor_generator(feats)
        objectness_logits, anchor_deltas = self.rpn_head(feats)

        # Reshape
        self.objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in objectness_logits
        ]
        self.anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, 4, x.shape[-2], x.shape[-1]).permute(0, 3, 4, 1, 2).flatten(1, -2)
            for x in anchor_deltas
        ]
        self.proposals = self.predict_proposals(self.anchors, self.objectness_logits, self.anchor_deltas, img_sizes)

    def forward_train(self, feats, img_sizes, labels):
        self._forward(feats, img_sizes)
        return self.proposals, self.get_losses(labels)

    def forward_eval(self, feats, img_sizes):
        self._forward(feats, img_sizes)
        return self.proposals

    def get_losses(self, labels):
        anchors = torch.cat(self.anchors)
        matches, match_flags = match_label(anchors, labels, self.matcher)
        match_flags = sample_pos_neg(match_flags, self.rpn_num, self.rpn_positive_fraction)

        cls_labels, reg_labels = [], []
        fg_flags, valid_flags = [], []
        for label, match, match_flag in zip(labels, matches, match_flags):
            fg_flag = match_flag > 0
            valid_flag = match_flag >= 0

            fg_flags.append(fg_flag)
            valid_flags.append(valid_flag)

            matched_label = label[match]

            cls_label = match_flag[valid_flag]

            box_label = matched_label[fg_flag, :4]
            reg_label = self.box_coder.encode_single(box_label, anchors[fg_flag])

            cls_labels.append(cls_label)
            reg_labels.append(reg_label)

        cls_labels = torch.cat(cls_labels)
        reg_labels = torch.cat(reg_labels)
        fg_flags = torch.stack(fg_flags)
        valid_flags = torch.stack(valid_flags)

        rpn_loc_loss = F.smooth_l1_loss(
            torch.cat(self.anchor_deltas, dim=1)[fg_flags], reg_labels, beta=0.0, reduction="sum"
        )
        rpn_cls_loss = F.binary_cross_entropy_with_logits(
            torch.cat(self.objectness_logits, dim=1)[valid_flags], cls_labels.float(), reduction="sum"
        )
        normalizer = self.rpn_num * len(labels)
        losses = {"rpn_cls": rpn_cls_loss / normalizer, "rpn_loc": rpn_loc_loss / normalizer}
        return losses

    @torch.no_grad()
    def predict_proposals(self, anchors, objectness_logits, anchor_deltas, img_sizes):
        N = anchor_deltas[0].shape[0]
        device = anchor_deltas[0].device
        batch_idx = torch.arange(N, device=device)

        # 1. For each feature map
        topk_proposals = []
        topk_scores = []
        level_ids = []

        for level_id, (anchors_i, logits_i, anchor_deltas_i) in enumerate(
            zip(anchors, objectness_logits, anchor_deltas)
        ):
            B = anchors_i.size(1)
            anchor_deltas_i = anchor_deltas_i.reshape(-1, B)
            # Expand anchors to shape (N*Hi*Wi*A, B)
            anchors_i = anchors_i.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)

            proposals_i = self.box_coder.decode_single(anchor_deltas_i, anchors_i)

            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals_i = proposals_i.view(N, -1, B)

            # for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
            num_proposals_i = min(proposals_i.shape[1], self.pre_nms_topk[0 if self.training else 1])

            logits_i, idx = logits_i.sort(descending=True, dim=1)
            topk_scores_i = logits_i.narrow(1, 0, num_proposals_i)
            topk_idx = idx.narrow(1, 0, num_proposals_i)

            # each is N x topk
            topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

            topk_proposals.append(topk_proposals_i)
            topk_scores.append(topk_scores_i)
            level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))

        # 2. Concat all levels together
        topk_scores = torch.cat(topk_scores, dim=1)
        topk_proposals = torch.cat(topk_proposals, dim=1)
        level_ids = torch.cat(level_ids, dim=0)

        # 3. For each image, run a per-level NMS, and choose topk results.
        results = []
        for n, img_size in enumerate(img_sizes):
            proposals = topk_proposals[n]
            scores = topk_scores[n]
            lvl = level_ids

            valid_mask = torch.isfinite(proposals).all(dim=1) & torch.isfinite(scores)
            if not valid_mask.all():
                if self.training:
                    raise FloatingPointError("Predicted boxes or scores contain Inf/NaN. Training has diverged.")
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                lvl = lvl[valid_mask]
            proposals = clip_boxes_to_image(proposals, img_size)

            # filter empty boxes
            keep = nonempty(proposals)
            if keep.sum().item() != proposals.shape[0]:
                proposals = proposals[keep]
                scores = scores[keep]
                lvl = lvl[keep]
            keep = batched_nms(proposals, scores, lvl, self.nms_threshold)
            keep = keep[: self.post_nms_topk[0 if self.training else 1]]

            proposals = proposals[keep]
            scores = scores[keep]
            results.append({"boxes": proposals, "scores": scores})
        return results
