import torch
import torch.nn as nn
import torch.nn.functional as F

from gvcore.model import GenericModule

from model.block import Normalize
from model.block.block import ConvModule


class ASPPV3pAlt(GenericModule):
    def __init__(self, cfg):
        super(ASPPV3pAlt, self).__init__(cfg)

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
                ConvModule(
                    self.in_channels,
                    self.inner_channels,
                    1 if dilation == 1 else 3,
                    padding=0 if dilation == 1 else dilation,
                    dilation=dilation,
                    bias=False,
                    norm=self.norm_layer(self.inner_channels),
                    activation=nn.LeakyReLU(inplace=True),
                )
                for dilation in self.dilations
            ]
        )

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.inner_channels,
                1,
                bias=False,
                norm=self.norm_layer(self.inner_channels),
                activation=nn.LeakyReLU(inplace=True),
            ),
        )
        self.aspp_fuse = ConvModule(len(self.dilations) * self.inner_channels, self.inner_channels, 1, bias=False,)
        self.image_pool_fuse = nn.Sequential(ConvModule(self.inner_channels, self.inner_channels, 1, bias=False,),)
        self.fuse_bn = self.norm_layer(self.inner_channels)

        self.lowlevel_path = ConvModule(
            self.lowlevel_in_channels,
            self.lowlevel_inner_channels,
            1,
            bias=False,
            norm=self.norm_layer(self.lowlevel_inner_channels),
            activation=nn.ReLU(inplace=True),
        )

        self.last_conv = nn.Sequential(
            ConvModule(
                self.inner_channels + self.lowlevel_inner_channels,
                self.inner_channels,
                3,
                padding=1,
                bias=False,
                norm=self.norm_layer(self.inner_channels),
                activation=nn.ReLU(inplace=True),
            ),
            ConvModule(
                self.inner_channels,
                self.inner_channels,
                3,
                padding=1,
                bias=False,
                norm=self.norm_layer(self.inner_channels),
                activation=nn.ReLU(inplace=False),
            ),
        )

    def forward(self, feats: torch.Tensor, lowlevel_feats: torch.Tensor) -> torch.Tensor:
        aspp_feats = [aspp(feats) for aspp in self.aspp]
        aspp_feats = torch.cat(aspp_feats, dim=1)
        aspp_feats = self.aspp_fuse(aspp_feats)

        gp_feats = self.image_pool(feats)
        gp_feats = self.image_pool_fuse(gp_feats)

        aspp_feats += gp_feats.repeat(1, 1, aspp_feats.shape[2], aspp_feats.shape[3])
        aspp_feats = self.fuse_bn(aspp_feats)
        aspp_feats = F.leaky_relu(aspp_feats)

        lowlevel_feats = self.lowlevel_path(lowlevel_feats)
        aspp_feats = F.interpolate(aspp_feats, size=lowlevel_feats.shape[2:], mode="bilinear", align_corners=True)
        out_feats = torch.cat((aspp_feats, lowlevel_feats), dim=1)

        out_feats = self.last_conv(out_feats)

        return out_feats
