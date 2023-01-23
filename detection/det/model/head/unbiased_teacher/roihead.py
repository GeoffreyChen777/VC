import torch
import torch.nn.functional as F

from model.head.roihead import ROIHead as _ROIHead
from model.utils import match_label, sample_pos_neg
from model.loss.functional import focal_loss


class ROIHead(_ROIHead):
    def __init__(self, cfg):
        super(ROIHead, self).__init__(cfg)

    def forward_train(self, feats, proposals, labels, ignore_loc=False):
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

            if label.shape[0] == 0:
                matched_label = torch.zeros((1, 5), device=self.device)[match]
            else:
                matched_label = label[match]

            matched_label[bg_flag, 4] = self.num_classes
            cls_label = matched_label[valid_flag, 4]
            cls_labels.append(cls_label.long())

            if not ignore_loc:
                box_label = matched_label[fg_flag, :4]
                reg_label = self.box_coder.encode_single(box_label, proposal_box[fg_flag])
                reg_labels.append(reg_label)

        # Forward head
        self.pooled_feats = self.pooler(feats, valid_proposals)
        self.box_feats = self.box_head(self.pooled_feats)
        self.cls_logits, self.box_deltas = self.box_predictor(self.box_feats)

        return self.get_losses(cls_labels, reg_labels, fg_flags, ignore_loc)

    def get_losses(self, cls_labels, reg_labels, fg_flags, ignore_loc=False):
        cls_labels = torch.cat(cls_labels)
        roi_cls_loss = focal_loss(self.cls_logits, cls_labels, gamma=1.5) / cls_labels.shape[0]

        if not ignore_loc:
            reg_labels = torch.cat(reg_labels)
            fg_flags = torch.cat(fg_flags)

            if self.box_deltas.shape[1] == 4:  # cls-agnostic regression
                box_deltas = self.box_deltas[fg_flags]
            else:
                box_deltas = self.box_deltas.view(-1, self.num_classes, 4)[fg_flags, cls_labels[fg_flags]]
            roi_loc_loss = F.smooth_l1_loss(box_deltas, reg_labels, beta=0.0, reduction="sum") / max(
                1, cls_labels.numel()
            )
            return {"roi_cls": roi_cls_loss, "roi_loc": roi_loc_loss}
        else:
            return {"roi_cls": roi_cls_loss}
