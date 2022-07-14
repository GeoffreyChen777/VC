import torch.nn as nn
from gvcore.model.utils import xavier_fill


class FasterRCNNHead(nn.Sequential):
    def __init__(self, fc_in_channels=12544, fc_inner_channels=1024, fc_nums=2):
        super(FasterRCNNHead, self).__init__()
        for i in range(fc_nums):
            if i == 0:
                self.add_module("flatten", nn.Flatten())
            fc = nn.Linear(fc_in_channels, fc_inner_channels)
            self.add_module("fc{}".format(i + 1), fc)
            self.add_module("fc_relu{}".format(i + 1), nn.ReLU())
            fc_in_channels = fc_inner_channels

        self._init_modules()

    def _init_modules(self):
        for layer in self:
            if isinstance(layer, nn.Linear):
                xavier_fill(layer)

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x


class FastRCNNOutputLayers(nn.Module):
    def __init__(self, in_channels=1024, num_classes=80, cls_agnostic_bbox_reg=False):
        super(FastRCNNOutputLayers, self).__init__()

        self.cls_score = nn.Linear(in_channels, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(in_channels, num_bbox_reg_classes * 4)

        self._init_modules()

    def _init_modules(self):
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for layer in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.cls_score(x), self.bbox_pred(x)
