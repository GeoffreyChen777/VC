import torch
import torch.nn.functional as F

from model.head.unbiased_teacher.roihead import ROIHead as _ROIHead
from model.utils import match_label, sample_pos_neg
from model.loss.functional import focal_loss


class ROIHead(_ROIHead):
    def __init__(self, cfg):
        super(ROIHead, self).__init__(cfg)

        self.temprature = cfg.model.roihead.temprature

    def forward_train(
        self, feats, proposals, labels, potential_labels=None, linear_feats=None, loc_uncertains=None,
    ):
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
        additional_logits_idxs = []
        reg_loc_uncertains = []
        cls_probs = []
        cls_bias_idx = 0

        for proposal_box, label, match, match_flag, potential_label, loc_uncertain in zip(
            proposal_boxes, labels, matches, match_flags, potential_labels, loc_uncertains
        ):
            nagree_flag = torch.ne(label[:, 4], potential_label[:, 4])
            prob_matrix = torch.zeros((label.shape[0], self.num_classes + 1), device=self.device)
            prob_matrix.scatter_(1, label[:, 4].long().unsqueeze(1), 1)
            prob_matrix[nagree_flag] = prob_matrix[nagree_flag].scatter_(
                1, label[nagree_flag, 4].long().unsqueeze(1), -1
            )
            prob_matrix[nagree_flag] = prob_matrix[nagree_flag].scatter_(
                1, potential_label[nagree_flag, 4].long().unsqueeze(1), -1
            )

            fg_flag = match_flag > 0
            bg_flag = match_flag == 0
            valid_flag = match_flag >= 0

            valid_proposals.append(proposal_box[valid_flag])
            fg_flags.append(fg_flag[valid_flag])

            if label.shape[0] == 0:
                matched_label = torch.zeros((1, 5), device=self.device)[match]
                matched_probs = torch.zeros((1, self.num_classes + 1), device=self.device, dtype=torch.float)[match]
                matched_loc_uncertain = torch.zeros((1, 4), device=self.device, dtype=torch.bool)[match]
            else:
                matched_label = label[match]
                matched_probs = prob_matrix[match]
                matched_loc_uncertain = loc_uncertain[match]

            additional_logits_idx = match.float() + cls_bias_idx
            cls_bias_idx += label.shape[0]
            additional_logits_idx[bg_flag] = -1
            additional_logits_idxs.append(additional_logits_idx[valid_flag])

            matched_probs[bg_flag] = 0
            matched_probs[bg_flag, -1] = 1
            cls_prob = matched_probs[valid_flag]
            cls_probs.append(cls_prob)

            matched_label[bg_flag, 4] = self.num_classes
            cls_label = matched_label[valid_flag, 4]
            cls_labels.append(cls_label.long())

            box_label = matched_label[fg_flag, :4]
            matched_loc_uncertain = matched_loc_uncertain[fg_flag]

            reg_label = self.box_coder.encode_single(box_label, proposal_box[fg_flag])
            reg_labels.append(reg_label)
            reg_loc_uncertains.append(matched_loc_uncertain)

        # Forward head
        self.pooled_feats = self.pooler(feats, valid_proposals)
        self.box_feats = self.box_head(self.pooled_feats)
        self.cls_logits, self.box_deltas = self.box_predictor(self.box_feats)

        # Generate virtual logits
        linear_feats = torch.cat(linear_feats)
        noise_resistant_logits = torch.mm(self.box_feats, F.normalize(linear_feats).t()) / self.temprature
        self.cls_logits = torch.cat((self.cls_logits, noise_resistant_logits), dim=1)

        return self.get_losses(cls_labels, reg_labels, fg_flags, cls_probs, additional_logits_idxs, reg_loc_uncertains)

    def get_losses(self, cls_labels, reg_labels, fg_flags, cls_probs, additional_logits_idxs, reg_loc_uncertains):
        cls_labels = torch.cat(cls_labels)
        cls_probs = torch.cat(cls_probs)
        if self.cls_logits.shape[1] > self.num_classes + 1:
            additional_logits_idxs = torch.cat(additional_logits_idxs).long()
            additional_probs = torch.zeros(
                (additional_logits_idxs.shape[0], self.cls_logits.shape[1] - self.num_classes - 1), device=self.device
            ).fill_(-1)

            flag = additional_logits_idxs >= 0
            additional_probs[flag].scatter_(dim=1, index=additional_logits_idxs[flag].unsqueeze(1), value=1)
            cls_probs = torch.cat((cls_probs, additional_probs), dim=1)
        roi_cls_loss = focal_loss(self.cls_logits, cls_probs, gamma=1.5) / cls_labels.shape[0]

        reg_loc_uncertains = torch.cat(reg_loc_uncertains)
        if reg_loc_uncertains.numel() == 0:
            roi_loc_loss = 0 * self.box_deltas.sum()
        else:
            reg_labels = torch.cat(reg_labels)
            fg_flags = torch.cat(fg_flags)

            if self.box_deltas.shape[1] == 4:  # cls-agnostic regression
                box_deltas = self.box_deltas[fg_flags]
            else:
                box_deltas = self.box_deltas.view(-1, self.num_classes, 4)[fg_flags, cls_labels[fg_flags]]
            roi_loc_loss = F.smooth_l1_loss(box_deltas, reg_labels, beta=0.0, reduction="none")

            loc_loss_xw_flag = (reg_loc_uncertains[:, 0] & reg_loc_uncertains[:, 2]).float()
            loc_loss_yh_flag = (reg_loc_uncertains[:, 1] & reg_loc_uncertains[:, 3]).float()
            loc_loss_flag = torch.stack((loc_loss_xw_flag, loc_loss_yh_flag, loc_loss_xw_flag, loc_loss_yh_flag)).t()
            roi_loc_loss = (roi_loc_loss * loc_loss_flag).sum()
            roi_loc_loss = roi_loc_loss / max(1, cls_labels.numel())

        return {"roi_cls": roi_cls_loss, "roi_loc": roi_loc_loss}
