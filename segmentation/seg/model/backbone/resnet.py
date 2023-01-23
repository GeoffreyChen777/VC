import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple

from gvcore.model import GenericModule

from model.block import Normalize
from model.block.block import ConvModule

__all__ = ["ResNet", "resnet50", "resnet101"]


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # TODO: In original MSRA resnet, stride is in 1x1, in pytorch stride is in 3x3 !
        self.conv1 = ConvModule(
            in_channels=inplanes, out_channels=width, kernel_size=1, stride=1, bias=False, norm=norm_layer(width)
        )
        self.conv2 = ConvModule(
            in_channels=width,
            out_channels=width,
            kernel_size=3,
            stride=stride,
            groups=groups,
            padding=dilation,
            dilation=dilation,
            bias=False,
            norm=norm_layer(width),
        )
        self.conv3 = ConvModule(
            in_channels=width,
            out_channels=planes * self.expansion,
            kernel_size=1,
            bias=False,
            norm=norm_layer(planes * self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(GenericModule):
    archs = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
    }  # type: Dict[int, Tuple[Bottleneck, Tuple[int, ...]]]

    def __init__(
        self,
        depth: int,
        in_channels: int = 3,
        stem_channels: int = 64,
        base_channels: int = 64,
        num_stages: int = 4,
        strides: Tuple[int] = (1, 2, 2, 2),
        dilations: Tuple[int] = (1, 1, 1, 1),
        contract_dilation: bool = False,
        multi_grid: Optional[List[int]] = None,
        deep_stem: bool = False,
        frozen_stages: int = -1,
        zero_init_residual: bool = False,
        norm_layer: Optional[str] = None,
        out_indices: List[int] = [1, 2, 3, 4],
        pretrained: Optional[str] = None,
        **kwargs,
    ) -> None:
        super(ResNet, self).__init__()

        assert depth in self.archs, f"Invalid depth {depth} for ResNet."
        assert num_stages >= 1 and num_stages <= 4, f"Invalid num_stages {num_stages} for ResNet."
        assert (
            len(strides) == len(dilations) == num_stages
        ), f"Length of strides {strides} and dilations {dilations} must match num_stages {num_stages}."
        assert max(out_indices) <= num_stages, f"Invalid out_indices {out_indices} for ResNet."

        # ========================
        # Params
        self.depth = depth
        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.strides = strides
        self.dilations = dilations
        self.contract_dilation = contract_dilation
        self.multi_grid = multi_grid
        self.deep_stem = deep_stem
        self.frozen_stages = frozen_stages
        self.zero_init_residual = zero_init_residual
        if norm_layer is None:
            norm_layer = "BN"
        self.norm_layer = Normalize(norm_layer)
        self.out_indices = out_indices
        self.pretrained = pretrained

        self.block, self.stage_block_nums = self.archs[depth]
        self.stage_block_nums = self.stage_block_nums[:num_stages]

        self.inplanes = self.stem_channels

        # ========================
        # Components
        self.stem = self._make_stem_layer()

        self.res_layers = []
        for i, block_num in enumerate(self.stage_block_nums):
            stride = strides[i]
            dilation = dilations[i]
            stage_multi_grid = self.multi_grid if i == len(self.stage_block_nums) - 1 else None

            planes = self.base_channels * 2 ** i
            res_layer = self._make_residual_layer(block_num, planes, stride, dilation, stage_multi_grid)
            self.res_layers.append(res_layer)
        self.res_layers = nn.Sequential(*self.res_layers)

        self._freeze_stages()
        self._init_modules()

    def _init_modules(self):
        if self.pretrained:
            self.load_state_dict(torch.load(self.pretrained))
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, self.block):
                        nn.init.constant_(m.conv3.norm.weight, 0)  # type: ignore[arg-type]

    def _make_stem_layer(self) -> nn.Sequential:
        if self.deep_stem:
            stem = nn.Sequential(
                ConvModule(
                    in_channels=self.in_channels,
                    out_channels=self.stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                    norm=self.norm_layer(self.stem_channels // 2),
                    activation=nn.ReLU(inplace=True),
                ),
                ConvModule(
                    in_channels=self.stem_channels // 2,
                    out_channels=self.stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    norm=self.norm_layer(self.stem_channels // 2),
                    activation=nn.ReLU(inplace=True),
                ),
                ConvModule(
                    in_channels=self.stem_channels // 2,
                    out_channels=self.stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    norm=self.norm_layer(self.stem_channels),
                    activation=nn.ReLU(inplace=True),
                ),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            stem = nn.Sequential(
                ConvModule(
                    in_channels=self.in_channels,
                    out_channels=self.stem_channels,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                    norm=self.norm_layer(self.stem_channels),
                    activation=nn.ReLU(inplace=True),
                ),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        return stem

    def _make_residual_layer(
        self, block_num: int, planes: int, stride: int, dilation: int, multi_grid: Optional[List[int]]
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * self.block.expansion:
            downsample = nn.Sequential(
                ConvModule(
                    in_channels=self.inplanes,
                    out_channels=planes * self.block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    norm=self.norm_layer(planes * self.block.expansion),
                ),
            )

        if multi_grid is None:
            if dilation > 1 and self.contract_dilation:
                first_dilation = dilation // 2
            else:
                first_dilation = dilation
        else:
            first_dilation = multi_grid[0]

        layers = []
        layers.append(
            self.block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                dilation=first_dilation,
                downsample=downsample,
                norm_layer=self.norm_layer,
            )
        )
        self.inplanes = planes * self.block.expansion
        for i in range(1, block_num):
            layers.append(
                self.block(
                    inplanes=self.inplanes,
                    planes=planes,
                    stride=1,
                    dilation=dilation if multi_grid is None else multi_grid[i],
                    norm_layer=self.norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            m = self.res_layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()

    def forward(self, x: Tensor) -> List[Tensor]:
        outs = []
        x = self.stem(x)

        if 0 in self.out_indices:
            outs.append(x)

        for i, layer in enumerate(self.res_layers):
            x = layer(x)
            if i + 1 in self.out_indices:
                outs.append(x)

        return outs

