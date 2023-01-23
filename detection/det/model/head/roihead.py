import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign, clip_boxes_to_image

from gvcore.utils.misc import attach_batch_idx
from gvcore.model import GenericModule

from utils.box import area

from model.head.fasterrcnn import FasterRCNNHead, FastRCNNOutputLayers
from model.anchor import BoxCoder, Matcher
from model.utils import match_label, sample_pos_neg


class ROIPooler(nn.Module):
    def __init__(self, output_size, scales, canonical_box_size=224, canonical_level=4):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

        self.level_poolers = nn.ModuleList(
            RoIAlign(output_size, spatial_scale=scale, sampling_ratio=0, aligned=True) for scale in scales
        )
        self.min_level = int(-(math.log2(scales[0])))
        self.max_level = int(-(math.log2(scales[-1])))
        self.canonical_level = canonical_level
        self.canonical_box_size = canonical_box_size
        self.level_assignments = None

    def assign_boxes_to_levels(self, batch_boxes):
        box_area = area(batch_boxes[:, 1:])
        box_sizes = torch.sqrt(box_area)
        # Eqn.(1) in FPN paper
        level_assignments = torch.floor(self.canonical_level + torch.log2(box_sizes / self.canonical_box_size + 1e-8))
        # clamp level to (min, max), in case the box size is too large or too small
        # for the available feature maps
        level_assignments = torch.clamp(level_assignments, min=self.min_level, max=self.max_level)
        return level_assignments.to(torch.int64) - self.min_level

    def forward(self, x, box_lists):
        num_level_assignments = len(self.level_poolers)
        box_lists = [torch.narrow(box, start=0, length=4, dim=1) for box in box_lists]
        batch_boxes = attach_batch_idx(box_lists)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], batch_boxes)

        level_assignments = self.assign_boxes_to_levels(batch_boxes)
        self.level_assignments = level_assignments

        num_boxes = batch_boxes.size(0)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros((num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device)

        for level, pooler in enumerate(self.level_poolers):
            inds = torch.nonzero(level_assignments == level, as_tuple=True)[0]
            cur_level_boxes = batch_boxes[inds]
            output.index_put_((inds,), pooler(x[level], cur_level_boxes))
        return output


class ROIHead(GenericModule):
    def __init__(self, cfg):
        super(ROIHead, self).__init__(cfg)

        # Params ==============
        self.in_feats = ("p2", "p3", "p4", "p5")
        self.num_classes = cfg.model.roihead.num_classes
        self.roi_num = 512
        self.roi_positive_fraction = 0.25

        # Components ==========
        self.box_head = FasterRCNNHead()
        self.box_predictor = FastRCNNOutputLayers(num_classes=self.num_classes)

        # Tools ===============
        self.pooler = ROIPooler(output_size=7, scales=[1 / x for x in (4, 8, 16, 32)])
        self.box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.matcher = Matcher(thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False)

        # Buffer ==============
        self.pooled_feats = None
        self.box_feats = None
        self.cls_logits = None
        self.box_deltas = None

    def forward_train(self, feats, proposals, labels):
        feats = [feats[f] for f in self.in_feats]

        # Add ground truth to proposals
        proposal_boxes = []
        for proposal, label in zip(proposals, labels):
            proposal_box = torch.cat((proposal["boxes"], label[:, :4]))
            proposal_boxes.append(proposal_box)

        matches, match_flags = match_label(proposal_boxes, labels, self.matcher)
        match_flags = sample_pos_neg(match_flags, self.roi_num, self.roi_positive_fraction)

        cls_labels, reg_labels = [], []
        fg_flags = []
        valid_proposals = []
        for proposal_box, label, match, match_flag in zip(proposal_boxes, labels, matches, match_flags):
            fg_flag = match_flag > 0
            bg_flag = match_flag == 0
            valid_flag = match_flag >= 0

            valid_proposals.append(proposal_box[valid_flag])
            fg_flags.append(fg_flag[valid_flag])

            matched_label = label[match]

            matched_label[bg_flag, 4] = self.num_classes
            cls_label = matched_label[valid_flag, 4]

            box_label = matched_label[fg_flag, :4]
            reg_label = self.box_coder.encode_single(box_label, proposal_box[fg_flag])

            cls_labels.append(cls_label.long())
            reg_labels.append(reg_label)

        cls_labels = torch.cat(cls_labels)
        reg_labels = torch.cat(reg_labels)
        fg_flags = torch.cat(fg_flags)

        # Forward head
        self.pooled_feats = self.pooler(feats, valid_proposals)
        self.box_feats = self.box_head(self.pooled_feats)
        self.cls_logits, self.box_deltas = self.box_predictor(self.box_feats)

        return self.get_losses(cls_labels, reg_labels, fg_flags)

    def get_losses(self, cls_labels, reg_labels, fg_flags):
        roi_cls_loss = F.cross_entropy(self.cls_logits, cls_labels, reduction="mean")

        if self.box_deltas.shape[1] == 4:  # cls-agnostic regression
            box_deltas = self.box_deltas[fg_flags]
        else:
            box_deltas = self.box_deltas.view(-1, self.num_classes, 4)[fg_flags, cls_labels[fg_flags]]
        roi_loc_loss = F.smooth_l1_loss(box_deltas, reg_labels, beta=0.0, reduction="sum") / max(1, cls_labels.numel())
        return {"roi_cls": roi_cls_loss, "roi_loc": roi_loc_loss}

    def forward_eval(self, feats, proposals, img_sizes):
        feats = [feats[f] for f in self.in_feats]

        proposal_boxes = [proposal["boxes"] for proposal in proposals]
        proposals_n = [proposal_box.shape[0] for proposal_box in proposal_boxes]

        self.pooled_feats = self.pooler(feats, proposal_boxes)
        self.box_feats = self.box_head(self.pooled_feats)
        self.cls_logits, self.box_deltas = self.box_predictor(self.box_feats)

        cls_scores = torch.softmax(self.cls_logits, dim=1)

        cls_scores = cls_scores.split(proposals_n)
        box_regs = self.box_deltas.split(proposals_n)

        detected_scores = []
        detected_boxes = []

        for cls_score, box_reg, proposal_box, img_size in zip(cls_scores, box_regs, proposal_boxes, img_sizes):
            box = self.box_coder.decode_single(box_reg, proposal_box)
            valid_mask = torch.isfinite(box).all(dim=1) & torch.isfinite(cls_score).all(dim=1)
            if not valid_mask.all():
                box = box[valid_mask]
                cls_score = cls_score[valid_mask]

            cls_score = torch.narrow(cls_score, dim=1, start=0, length=self.num_classes)

            num_box_each_position = box.shape[1] // 4
            box = box.reshape(-1, 4)
            box = clip_boxes_to_image(box, img_size)
            box = box.view(-1, num_box_each_position, 4)

            detected_scores.append(cls_score)
            detected_boxes.append(box)

        return detected_scores, detected_boxes
