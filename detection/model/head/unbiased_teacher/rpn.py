import torch
import torch.nn.functional as F

from model.utils import match_label, sample_pos_neg
from model.head.rpn import RPN as _RPN


class RPN(_RPN):
    def __init__(self, cfg):
        super(RPN, self).__init__(cfg)
        self.rpn_positive_fraction = 0.25

    def forward_train(self, feats, img_sizes, labels, ignore_loc=False):
        self._forward(feats, img_sizes)
        return self.proposals, self.get_losses(labels, ignore_loc)

    def get_losses(self, labels, ignore_loc=False):
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

            cls_label = match_flag[valid_flag]
            cls_labels.append(cls_label)
            if not ignore_loc:
                if label.shape[0] == 0:
                    matched_label = torch.zeros((1, 5), device=self.device)[match]
                else:
                    matched_label = label[match]

                box_label = matched_label[fg_flag, :4]
                reg_label = self.box_coder.encode_single(box_label, anchors[fg_flag])
                reg_labels.append(reg_label)

        cls_labels = torch.cat(cls_labels)
        fg_flags = torch.stack(fg_flags)
        valid_flags = torch.stack(valid_flags)

        normalizer = self.rpn_num * len(labels)

        rpn_cls_loss = F.binary_cross_entropy_with_logits(
            torch.cat(self.objectness_logits, dim=1)[valid_flags], cls_labels.float(), reduction="sum"
        )
        if not ignore_loc:
            reg_labels = torch.cat(reg_labels)
            rpn_loc_loss = F.smooth_l1_loss(
                torch.cat(self.anchor_deltas, dim=1)[fg_flags], reg_labels, beta=0.0, reduction="sum"
            )
            losses = {"rpn_cls": rpn_cls_loss / normalizer, "rpn_loc": rpn_loc_loss / normalizer}
        else:
            losses = {"rpn_cls": rpn_cls_loss / normalizer}
        return losses
