import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 添加BIFPN模块引用到的东西
class DepthWiseSeparableConvModule(nn.Module):
    """ DepthWise Separable Convolution with BatchNorm and ReLU activation """
    def __init__(self, in_channels, out_channels, bath_norm=True, act="silu", bias=False):
        super(DepthWiseSeparableConvModule, self).__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.conv_dw = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                 padding=1, groups=in_channels, bias=False)
        self.conv_pw = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                 padding=0, bias=bias)

        self.bn = None if not bath_norm else \
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        # self.act = None if not relu else Swish()

        self.act = get_activation(act, inplace=True)


    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class ConvModule(nn.Module):
    """ Regular Convolution with BatchNorm """
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, act="silu"):
        super(ConvModule, self).__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)

        # self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

# 设置激活函数
def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


# class Swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)

class MaxPool2dSamePad(nn.MaxPool2d):
    """ TensorFlow-like 2D Max Pooling with same padding """

    PAD_VALUE: float = -float('inf')

    def __init__(self, kernel_size: int, stride=1, padding=0,
                 dilation=1, ceil_mode=False, count_include_pad=True):
        assert padding == 0, 'Padding in MaxPool2d Same Padding should be zero'

        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        padding = (padding, padding)
        dilation = (dilation, dilation)

        super(MaxPool2dSamePad, self).__init__(kernel_size, stride, padding,
                                               dilation, ceil_mode, count_include_pad)

    def forward(self, x):
        h, w = x.size()[-2:]

        pad_h = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] + \
                (self.kernel_size[0] - 1) * self.dilation[0] + 1 - h
        pad_w = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] + \
                (self.kernel_size[1] - 1) * self.dilation[1] + 1 - w

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                          pad_h - pad_h // 2], value=self.PAD_VALUE)

        x = F.max_pool2d(x, self.kernel_size, self.stride,
                         self.padding, self.dilation, self.ceil_mode)
        return x