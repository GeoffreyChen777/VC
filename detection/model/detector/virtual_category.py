import torch
from torchvision.ops.boxes import batched_nms
from dataset.transforms.functional import transform_label_by_size

from gvcore.utils.structure import TensorDict
from gvcore.utils.types import TDataList, TTensor, TTensorTuple
from gvcore.model import GenericModule

from utils.box import pairwise_iou

from model.detector.ts import TSDetector
from model.detector.unbiased_teacher import UBTDetector
from model.head.virtual_category.roihead import ROIHead

from dataset.transforms import BatchApply


class VCDetector(UBTDetector):
    def __init__(self, cfg):
        super(VCDetector, self).__init__(cfg)
        self.roi_heads = ROIHead(cfg)

    def forward_train(self, data_list, labeled=False):
        imgs = torch.stack([data.img for data in data_list], dim=0)
        labels = [data.label for data in data_list]
        img_sizes = [data.meta.cur_size for data in data_list]

        feats = self.backbone(imgs)
        feats = self.fpn(feats)
        self.feats = feats

        proposals, rpn_loss_dict = self.rpn(feats, img_sizes, labels, not labeled)

        if not labeled:
            potential_labels = [data.potential_label for data in data_list]
            loc_uncertains = [data.loc_uncertain for data in data_list]
            linear_feats = [data.label_linear_feat for data in data_list]
        else:
            potential_labels = [data.label for data in data_list]
            loc_uncertains = [
                torch.ones((data.label.shape[0], 4), device=self.device, dtype=torch.bool) for data in data_list
            ]
            linear_feats = [torch.empty((0, 1024), device=self.device) for _ in data_list]

        roi_loss_dict = self.roi_heads(feats, proposals, labels, potential_labels, linear_feats, loc_uncertains)

        loss_dict = {}
        loss_dict.update(rpn_loss_dict)
        loss_dict.update(roi_loss_dict)
        return loss_dict

    def forward_eval(self, data_list, proposals=None):
        imgs = torch.stack([data.img for data in data_list], dim=0)
        img_sizes = [data.meta.cur_size for data in data_list]

        feats = self.backbone(imgs)
        feats = self.fpn(feats)
        self.feats = feats

        if proposals is None:
            proposals = self.rpn(feats, img_sizes)
            detected_scores, detected_boxes = self.roi_heads(feats, proposals, img_sizes)

            detected_bboxes = []
            for score, box in zip(detected_scores, detected_boxes):
                filter_mask = torch.gt(score, self.score_threshold)
                cls = torch.nonzero(filter_mask, as_tuple=True)[1]
                box = box[filter_mask]
                score = score[filter_mask]

                keep = batched_nms(box, score, cls, self.nms_threshold)
                keep = keep[: self.max_detections]
                box, score, cls = box[keep], score[keep], cls[keep]
                detected_bbox = torch.cat((box, cls.view(-1, 1), score.view(-1, 1)), dim=1)
                detected_bboxes.append(detected_bbox)
        else:
            detected_scores, detected_boxes = self.roi_heads(feats, proposals, img_sizes)

            detected_bboxes = []
            for score, box in zip(detected_scores, proposals):
                score = torch.cat((score, 1 - score.sum(dim=1, keepdim=True)), dim=1)
                if score.numel() > 0:
                    score, cls = torch.max(score, dim=1)
                else:
                    score, cls = torch.empty((0, 1), device=self.device), torch.empty((0, 1), device=self.device)
                detected_bbox = torch.cat((box["boxes"], cls.view(-1, 1), score.view(-1, 1)), dim=1)
                detected_bboxes.append(detected_bbox)

        return detected_bboxes


class VCTSDetector(TSDetector):
    """
    Semi-supervised Teacher-Student detector with temporal stability varify and VC loss.
    """

    def __init__(self, cfg, detector_class: GenericModule):
        super(VCTSDetector, self).__init__(cfg, detector_class)

        self.flip = BatchApply(**{"random_horizontal_flip": [1]})

        self.ema_detector.roi_heads.box_predictor.cls_score.register_forward_hook(self._hook_linear_feats)
        self.linear_feats = None

        self.num_classes = cfg.model.roihead.num_classes
        self.loc_t = cfg.model.roihead.loc_t

        self.history = TensorDict()

    def _hook_linear_feats(self, hooked_module, input, output):
        self.linear_feats = input[0].detach()

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

        # 1. Pseudo labeling and create PC set
        pseudo_labels = self.ema_detector(weak_data_list_u)
        ignore_loc = False
        for p_label, data in zip(pseudo_labels, strong_data_list_u):
            p_label = p_label[p_label[:, 5] > self.pseudo_t]

            if data.meta.id in self.history:
                h_label = self.history[data.meta.id].to(self.device)
                h_label = transform_label_by_size(h_label, data.meta.ori_size, data.meta.cur_size, data.meta.flip)
                ignore_loc = False
            else:
                h_label = p_label.clone()
                ignore_loc = True

            self.history[data.meta.id] = transform_label_by_size(
                p_label, data.meta.cur_size, data.meta.ori_size, data.meta.flip
            )

            label, potential_label, loc_uncertain = self._compare(p_label, h_label, ignore_loc=ignore_loc)

            data.label = label
            data.potential_label = potential_label
            data.loc_uncertain = loc_uncertain

        # 2. Extract feature vector as virtual weights
        # 2.1 Flip boxes
        proposals = []
        for data in strong_data_list_u:
            w = data.img.shape[2]
            proposal = data.label.clone()
            proposal[:, [0, 2]] = w - proposal[:, [2, 0]]
            proposals.append({"boxes": proposal})
        # 2.2 Extract features
        self.ema_detector(self.weak_aug(self.flip(data_list_u)), proposals=proposals)
        linear_feat_list = self.linear_feats.split([data.label.shape[0] for data in strong_data_list_u])

        for data, linear_feat in zip(strong_data_list_u, linear_feat_list):
            data.label_linear_feat = linear_feat.detach()

        # 3. Forward unlabeled data
        loss_dict_u = self.detector(strong_data_list_u, labeled=False)

        return loss_dict_u
