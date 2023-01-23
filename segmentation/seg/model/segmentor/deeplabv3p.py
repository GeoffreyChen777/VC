from addict import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from gvcore.utils.types import TDataList
from gvcore.model import GenericModule

from model.backbone import make_backbone
from model.head.aspp import ASPPV3p


class DeeplabV3p(GenericModule):
    def __init__(self, cfg):
        super(DeeplabV3p, self).__init__(cfg)

        # Components ========
        self.backbone = make_backbone(cfg.model.backbone)

        self.decoder = ASPPV3p(cfg.model.aspp)

        self.dropout = nn.Dropout2d(cfg.model.dropout)
        self.classifier = nn.Conv2d(self.decoder.inner_channels, cfg.model.num_classes, 1)

    def forward_generic(self, data_list: TDataList) -> torch.Tensor:
        imgs = torch.stack([data.img for data in data_list], dim=0)

        feat_list = self.backbone(imgs)
        feats = self.decoder(feats=feat_list[-1], lowlevel_feats=feat_list[0])

        logits = self.classifier(self.dropout(feats))
        logits = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

        return logits

    def forward_train(self, data_list: TDataList) -> Dict:
        labels = torch.stack([data.label for data in data_list], dim=0)
        logits = self.forward_generic(data_list)
        return self.get_losses(logits, labels)

    def get_losses(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict:
        loss_dict = Dict()
        loss = F.cross_entropy(logits, labels.long().squeeze(1), ignore_index=255, reduction="none")
        loss_mask = labels.squeeze(1) != 255
        loss = loss * loss_mask.float()
        loss = loss.sum() / (loss_mask.float().sum() + 1e-10)

        loss_dict.loss = loss

        return loss_dict

    def forward_eval(self, data_list: TDataList):
        logits = self.forward_generic(data_list)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        return Dict(pred_list=preds)
