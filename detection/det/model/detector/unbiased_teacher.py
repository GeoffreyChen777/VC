import torch

from gvcore.utils.types import TDataList

from model.detector.fasterrcnn import FasterRCNN
from model.head.unbiased_teacher.rpn import RPN
from model.head.unbiased_teacher.roihead import ROIHead


class UBTDetector(FasterRCNN):
    def __init__(self, cfg):
        super(UBTDetector, self).__init__(cfg)
        self.rpn = RPN(cfg)
        self.roi_heads = ROIHead(cfg)

    def forward_train(self, data_list: TDataList, labeled: bool = False):
        imgs = torch.stack([data.img for data in data_list], dim=0)
        labels = [data.label for data in data_list]
        img_sizes = [data.meta.cur_size for data in data_list]

        feats = self.backbone(imgs)
        feats = self.fpn(feats)
        self.feats = feats

        proposals, rpn_loss_dict = self.rpn(feats, img_sizes, labels, not labeled)
        roi_loss_dict = self.roi_heads(feats, proposals, labels, not labeled)

        loss_dict = {}
        loss_dict.update(rpn_loss_dict)
        loss_dict.update(roi_loss_dict)
        return loss_dict
