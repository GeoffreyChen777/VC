import torch
from model.backbone.fpn import FPN, LastLevelMaxPool
from torchvision.ops import batched_nms

from gvcore.utils.types import TDataList
from gvcore.model import GenericModule

from model.backbone import make_backbone
from model.head.rpn import RPN
from model.head.roihead import ROIHead


class FasterRCNN(GenericModule):
    def __init__(self, cfg):
        super(FasterRCNN, self).__init__(cfg)

        # Components ========
        self.backbone = make_backbone(cfg)
        base_inc = self.backbone.inplanes // 8
        fpn_in_channels_list = [base_inc * 2 ** (i - 1) for i in range(1, 5)]
        self.fpn = FPN(
            in_features=["res2", "res3", "res4", "res5"],
            in_channels_list=fpn_in_channels_list,
            out_channels=256,
            strides=[4, 8, 16, 32],
            top_block=LastLevelMaxPool(),
        )
        self.rpn = RPN(cfg)
        self.roi_heads = ROIHead(cfg)

        # == Params ====================
        self.score_threshold = cfg.model.score_threshold
        self.nms_threshold = cfg.model.nms_threshold
        self.max_detections = cfg.model.max_detections

        self.feats = None

    def forward_train(self, data_list: TDataList):
        imgs = torch.stack([data.img for data in data_list], dim=0)
        labels = [data.label for data in data_list]
        img_sizes = [data.meta.cur_size for data in data_list]

        feats = self.backbone(imgs)
        feats = self.fpn(feats)
        self.feats = feats

        proposals, rpn_loss_dict = self.rpn(feats, img_sizes, labels)
        roi_loss_dict = self.roi_heads(feats, proposals, labels)

        loss_dict = {}
        loss_dict.update(rpn_loss_dict)
        loss_dict.update(roi_loss_dict)
        return loss_dict

    def forward_eval(self, data_list: TDataList):
        imgs = torch.stack([data.img for data in data_list], dim=0)
        img_sizes = [data.meta.cur_size for data in data_list]

        feats = self.backbone(imgs)
        feats = self.fpn(feats)
        self.feats = feats

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

        return detected_bboxes
