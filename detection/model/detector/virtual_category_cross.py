import torch
from torchvision.ops.boxes import batched_nms
from dataset.transforms.functional import transform_label_by_size

from gvcore.utils.structure import TensorDict
from gvcore.utils.misc import split_ab, merge_loss_dict
from gvcore.utils.types import TDataList, TTensor, TTensorTuple
from gvcore.model import GenericModule

from utils.box import pairwise_iou

from model.detector.ts_cross import TSCrossDetector

from dataset.transforms import BatchApply


class VCCrossTSDetector(TSCrossDetector):
    """
    Semi-supervised Teacher-Student detector with temporal stability varify and VC loss.
    """

    def __init__(self, cfg, detector_class: GenericModule):
        super(VCCrossTSDetector, self).__init__(cfg, detector_class)

        self.flip = BatchApply(**{"random_horizontal_flip": [1]})

        self.ema_detector_a.roi_heads.box_predictor.cls_score.register_forward_hook(self._hook_linear_feats_a)
        self.ema_detector_b.roi_heads.box_predictor.cls_score.register_forward_hook(self._hook_linear_feats_b)
        self.linear_feats_a = None
        self.linear_feats_b = None

        self.num_classes = cfg.model.roihead.num_classes
        self.loc_t = cfg.model.roihead.loc_t

    def _hook_linear_feats_a(self, hooked_module, input, output):
        self.linear_feats_a = input[0].detach()

    def _hook_linear_feats_b(self, hooked_module, input, output):
        self.linear_feats_b = input[0].detach()

    def _compare(self, p_label: TTensor, h_label: TTensor, ignore_loc: bool = True) -> TTensorTuple:
        merged_label = torch.cat((p_label, h_label))
        keep = batched_nms(merged_label[:, :4], merged_label[:, 5], merged_label[:, 4], self.ema_detector.nms_threshold)

        if merged_label.shape[0] == 0:
            potential_label = merged_label.clone()
            return merged_label[keep], potential_label[keep], merged_label[keep, :4]
        else:
            iou = pairwise_iou(merged_label[:, :4], merged_label[:, :4])
            iou.fill_diagonal_(-1)

            max_iou, max_idx = torch.max(iou, dim=1)
            match_label = merged_label[max_idx]

            w = merged_label[:, 2] - merged_label[:, 0]
            h = merged_label[:, 3] - merged_label[:, 1]

            if not ignore_loc:
                loc_diff = merged_label[:, :4] - match_label[:, :4]
                loc_diff_ratio = loc_diff.abs() / torch.stack((w, h, w, h)).t()
                loc_uncertain = torch.lt(loc_diff_ratio, self.loc_t)
            else:
                loc_uncertain = torch.zeros((merged_label.shape[0], 4), device=self.device, dtype=torch.bool)

            match_label[max_iou < 0.5, 4] = self.num_classes
            potential_label = merged_label.clone()
            potential_label[:, 4] = match_label[:, 4]
            return merged_label[keep], potential_label[keep], loc_uncertain[keep]

    def forward_train_u(self, data_list_u: TDataList):

        self._momentum_update(m=self.m)

        weak_data_list_u = self.weak_aug(data_list_u)
        strong_data_list_u = self.strong_aug(data_list_u)
        weak_data_list_u_a, weak_data_list_u_b = split_ab(weak_data_list_u)
        strong_data_list_u_a, strong_data_list_u_b = split_ab(strong_data_list_u)

        # 1. Pseudo labeling
        pseudo_labels_a = self.ema_detector_b(weak_data_list_u_a)
        pseudo_labels_b = self.ema_detector_a(weak_data_list_u_b)

        for p_label, data in zip(pseudo_labels_a, strong_data_list_u_a):
            p_label = p_label[p_label[:, 5] > self.pseudo_t]
            data.label = p_label
        for p_label, data in zip(pseudo_labels_b, strong_data_list_u_b):
            p_label = p_label[p_label[:, 5] > self.pseudo_t]
            data.label = p_label

        # 2. Create PC set
        # 2.1 Flip boxes
        proposals_a = []
        for data in strong_data_list_u_a:
            w = data.img.shape[2]
            proposal = data.label.clone()
            proposal[:, [0, 2]] = w - proposal[:, [2, 0]]
            proposals_a.append({"boxes": proposal})
        # 2.2 Get potential labels and extract features
        potential_labels_a = self.ema_detector_a(self.flip(weak_data_list_u_a), proposals=proposals_a)
        linear_feat_list_a = self.linear_feats_a.split([data.label.shape[0] for data in strong_data_list_u_a])
        for data, potential_label, linear_feat in zip(strong_data_list_u_a, potential_labels_a, linear_feat_list_a):
            data.label_linear_feat = linear_feat.detach()
            w = data.img.shape[2]
            potential_label[:, [0, 2]] = w - potential_label[:, [2, 0]]
            data.potential_label = potential_label

            w = data.label[:, 2] - data.label[:, 0]
            h = data.label[:, 3] - data.label[:, 1]

            loc_diff = data.label[:, :4] - potential_label[:, :4]
            loc_diff_ratio = loc_diff.abs() / torch.stack((w, h, w, h)).t()
            loc_uncertain = torch.lt(loc_diff_ratio, self.loc_t)

            data.loc_uncertain = loc_uncertain

        potential_labels_b = self.ema_detector_b(self.flip(weak_data_list_u_b), proposals=proposals_a)
        linear_feat_list_b = self.linear_feats_b.split([data.label.shape[0] for data in strong_data_list_u_b])
        for data, potential_label, linear_feat in zip(strong_data_list_u_b, potential_labels_b, linear_feat_list_b):
            data.label_linear_feat = linear_feat.detach()
            w = data.img.shape[2]
            potential_label[:, [0, 2]] = w - potential_label[:, [2, 0]]
            data.potential_label = potential_label

            w = data.label[:, 2] - data.label[:, 0]
            h = data.label[:, 3] - data.label[:, 1]

            loc_diff = data.label[:, :4] - potential_label[:, :4]
            loc_diff_ratio = loc_diff.abs() / torch.stack((w, h, w, h)).t()
            loc_uncertain = torch.lt(loc_diff_ratio, self.loc_t)

            data.loc_uncertain = loc_uncertain

        # 3. Forward unlabeled data
        loss_dict_u_a = self.detector_a(strong_data_list_u_a, labeled=False)
        loss_dict_u_b = self.detector_a(strong_data_list_u_b, labeled=False)

        loss_dict_u = merge_loss_dict(loss_dict_u_a, loss_dict_u_b)
        return loss_dict_u
