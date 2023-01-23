import torch
import torch.nn as nn
import torch.nn.functional as F

from gvcore.model import GenericModule

from model.block import Normalize
from model.block.block import ConvModule, DepthwiseSeparableConvModule


class ASPPV3p(GenericModule):
    def __init__(self, cfg):
        super(ASPPV3p, self).__init__(cfg)

        # Params ==============
        self.in_channels = cfg.in_channels
        self.inner_channels = cfg.inner_channels

        self.lowlevel_in_channels = cfg.lowlevel_in_channels
        self.lowlevel_inner_channels = cfg.lowlevel_inner_channels

        self.dilations = cfg.dilations
        self.norm_layer = Normalize(cfg.norm_layer)

        # Components ==========
        self.aspp = nn.ModuleList(
            [
                DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.inner_channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    dw_norm=self.norm_layer(self.in_channels),
                    pw_norm=self.norm_layer(self.inner_channels),
                    activation=nn.ReLU(inplace=True),
                )
                for dilation in self.dilations
            ]
        )
        self.aspp[0] = ConvModule(
            self.in_channels,
            self.inner_channels,
            1 if self.dilations[0] == 1 else 3,
            dilation=self.dilations[0],
            padding=0 if self.dilations[0] == 1 else self.dilations[0],
            bias=False,
            norm=self.norm_layer(self.inner_channels),
            activation=nn.ReLU(inplace=True),
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.inner_channels,
                1,
                bias=False,
                norm=self.norm_layer(self.inner_channels),
                activation=nn.ReLU(inplace=True),
            ),
        )
        self.aspp_fuse = ConvModule(
            (len(self.dilations) + 1) * self.inner_channels,
            self.inner_channels,
            3,
            padding=1,
            bias=False,
            norm=self.norm_layer(self.inner_channels),
            activation=nn.ReLU(inplace=True),
        )

        self.lowlevel_path = ConvModule(
            self.lowlevel_in_channels,
            self.lowlevel_inner_channels,
            1,
            bias=False,
            norm=self.norm_layer(self.lowlevel_inner_channels),
            activation=nn.ReLU(inplace=True),
        )

        self.last_conv = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.inner_channels + self.lowlevel_inner_channels,
                self.inner_channels,
                3,
                padding=1,
                bias=False,
                dw_norm=self.norm_layer(self.inner_channels + self.lowlevel_inner_channels),
                pw_norm=self.norm_layer(self.inner_channels),
                activation=nn.ReLU(inplace=True),
            ),
            DepthwiseSeparableConvModule(
                self.inner_channels,
                self.inner_channels,
                3,
                padding=1,
                bias=False,
                dw_norm=self.norm_layer(self.inner_channels),
                pw_norm=self.norm_layer(self.inner_channels),
                activation=nn.ReLU(inplace=True),
            ),
        )

    def forward(self, feats: torch.Tensor, lowlevel_feats: torch.Tensor) -> torch.Tensor:
        aspp_feats = []
        gp_feats = self.image_pool(feats)
        aspp_feats.append(F.interpolate(gp_feats, size=feats.shape[2:], mode="bilinear", align_corners=False))
        aspp_feats.extend([aspp(feats) for aspp in self.aspp])

        aspp_feats = torch.cat(aspp_feats, dim=1)
        aspp_feats = self.aspp_fuse(aspp_feats)

        lowlevel_feats = self.lowlevel_path(lowlevel_feats)
        aspp_feats = F.interpolate(aspp_feats, size=lowlevel_feats.shape[2:], mode="bilinear", align_corners=False)
        out_feats = torch.cat((aspp_feats, lowlevel_feats), dim=1)

        out_feats = self.last_conv(out_feats)

        return out_feats
