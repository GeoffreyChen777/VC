import torch
import torch.nn as nn
import torch.nn.functional as F

from gvcore.model.utils import kaiming_init, constant_init


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        norm=None,
        activation=None,
    ):
        super(ConvModule, self).__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode
        )

        self.norm = norm
        self.activation = activation

        # self.init_weights()

    def init_weights(self):
        if not hasattr(self.conv, "init_weights"):
            if isinstance(self.activation, nn.LeakyReLU):
                nonlinearity = "leaky_relu"
                a = self.activation.negative_slope
            else:
                nonlinearity = "relu"
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.norm is not None:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        dw_norm=None,
        pw_norm=None,
        activation=None,
        **kwargs
    ):
        super(DepthwiseSeparableConvModule, self).__init__()
        self.depthwise_conv = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            norm=dw_norm,
            activation=activation,
        )
        self.pointwise_conv = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
            norm=pw_norm,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
