import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .darknet import CSPDarknet
from .module import DepthWiseSeparableConvModule as DWSConv
from .module import MaxPool2dSamePad
from .module import get_activation
# from .CBAM import *
from .module import ConvModule

class YOLOBIAFPN(nn.Module):

    EPS: float = 1e-04
    REDUCTION_RATIO: int = 2


    def __init__(self,
                 depth,
                 width,
                 in_features=("dark3", "dark4", "dark5"),
                 in_channels=[256, 512, 1024],
                 # in_features=("dark2", "dark3", "dark4", "dark5"),
                 # in_channels=[128, 256, 512, 1024],
                 depthwise=False,  # 深度分离卷积
                 act="silu",
                 time=1,
                 # n_channels
    ):
        super().__init__()
        # 主干
        self.time = time
        self.backbone = None
        if self.time == 1:
            self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act="silu")
        self.in_features = in_features
        self.in_channels = in_channels

        self.act = get_activation(act, inplace=True)
        # self.act = Swish()


        # 分割线------------------------------------------
        # dark2，dark3，dark4，dark5
        # self.conv_3_td = DWSConv(in_channels[1] * width, in_channels[1] * width, act="silu")
        # self.conv_4_td = DWSConv(in_channels[2] * width, in_channels[2] * width, act="silu")
        #
        # self.weights_3_td = nn.Parameter(torch.ones(2))
        # self.weights_4_td = nn.Parameter(torch.ones(2))
        #
        # self.conv_2_out = DWSConv(in_channels[0] * width, in_channels[0] * width, act="silu")
        # self.conv_3_out = DWSConv(in_channels[1] * width, in_channels[1] * width, act="silu")
        # self.conv_4_out = DWSConv(in_channels[2] * width, in_channels[2] * width, act="silu")
        # self.conv_5_out = DWSConv(in_channels[3] * width, in_channels[3] * width, act="silu")
        #
        # self.weights_2_out = nn.Parameter(torch.ones(2))
        # self.weights_3_out = nn.Parameter(torch.ones(3))
        # self.weights_4_out = nn.Parameter(torch.ones(3))
        # self.weights_5_out = nn.Parameter(torch.ones(2))
        #
        # # self.upsample = lambda x: F.interpolate(x, scale_factor=self.REDUCTION_RATIO)
        # self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # self.downsample = MaxPool2dSamePad(self.REDUCTION_RATIO + 1, self.REDUCTION_RATIO)
        #
        # self.conv_5_to_4 = OneConv(in_channels[3] * width, in_channels[2] * width, ksize=1, stride=1)
        # self.conv_4_to_3 = OneConv(in_channels[2] * width, in_channels[1] * width, ksize=1, stride=1)
        # self.conv_3_to_2 = OneConv(in_channels[1] * width, in_channels[0] * width, ksize=1, stride=1)
        # self.conv_2_to_3 = OneConv(in_channels[0] * width, in_channels[1] * width, ksize=1, stride=1)
        # self.conv_3_to_4 = OneConv(in_channels[1] * width, in_channels[2] * width, ksize=1, stride=1)
        # self.conv_4_to_5 = OneConv(in_channels[2] * width, in_channels[3] * width, ksize=1, stride=1)
        #
        # self.cbam_5_to_4 = CBAM(in_channels[2] * width, in_channels[2] * width)
        # self.cbam_4_to_3 = CBAM(in_channels[1] * width, in_channels[1] * width)
        # self.cbam_3_to_2 = CBAM(in_channels[0] * width, in_channels[0] * width)
        # self.cbam_2_to_3 = CBAM(in_channels[1] * width, in_channels[1] * width)
        # self.cbam_3_to_4 = CBAM(in_channels[2] * width, in_channels[2] * width)
        # ---------------------------------------------------------


        # 分割线--------------------------------------------
        # dark3，dark4，dark5-------------------------------
        # self.conv_4_td = DWSConv(in_channels[1] * width, in_channels[1] * width, act="silu")
        # self.conv_3_out = DWSConv(in_channels[0] * width, in_channels[0] * width, act="silu")
        # self.conv_4_out = DWSConv(in_channels[1] * width, in_channels[1] * width, act="silu")
        # self.conv_5_out = DWSConv(in_channels[2] * width, in_channels[2] * width, act="silu")
        self.conv_4_td = DWSConv(in_channels[1] * width, in_channels[1] * width)
        self.conv_3_out = DWSConv(in_channels[0] * width, in_channels[0] * width)
        self.conv_4_out = DWSConv(in_channels[1] * width, in_channels[1] * width)
        self.conv_5_out = DWSConv(in_channels[2] * width, in_channels[2] * width)

        self.weights_4_td = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.weights_3_out = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.weights_4_out = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.weights_5_out = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)


        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.downsample = MaxPool2dSamePad(self.REDUCTION_RATIO + 1, self.REDUCTION_RATIO)

        self.conv_5_to_4 = ConvModule(in_channels[2] * width, in_channels[1] * width)
        self.conv_4_to_3 = ConvModule(in_channels[1] * width, in_channels[0] * width)
        self.conv_3_to_4 = ConvModule(in_channels[0] * width, in_channels[1] * width)
        self.conv_4_to_5 = ConvModule(in_channels[1] * width, in_channels[2] * width)




        # self.cbam_5_to_4 = CBAM(in_channels[1] * width, in_channels[1] * width)
        # self.cbam_4_to_3 = CBAM(in_channels[0] * width, in_channels[0] * width)
        # self.cbam_3_to_4 = CBAM(in_channels[1] * width, in_channels[1] * width)
        # -----------------------------------------



    def forward(self, input):

        # dark2，dark3，dark4，dark5-------------------------------
        # if self.backbone:
        #     out_features = self.backbone(input)  # 输入进backbone
        #     input = [out_features[f] for f in self.in_features]
        #
        # p_2, p_3, p_4, p_5 = input
        # p_3_2 = None
        # p_4_2 = None
        #
        # p_3_in = p_3 if p_3_2 is None else p_3_2
        # p_4_in = p_4 if p_4_2 is None else p_4_2
        #
        #
        # f_5_to_4 = self.conv_5_to_4(p_5)
        # p_4_td = self.conv_4_td(
        #     self._fuse_features(
        #         weights=self.weights_4_td,
        #         features=[p_4, self.upsample(f_5_to_4)]
        #     )
        # )
        # cbam_4_td = self.cbam_5_to_4(p_4_td)
        #
        # f_4_to_3 = self.conv_4_to_3(cbam_4_td)
        # p_3_td = self.conv_3_td(
        #     self._fuse_features(
        #         weights=self.weights_3_td,
        #         features=[p_3, self.upsample(f_4_to_3)]
        #     )
        # )
        # cbam_3_td = self.cbam_4_to_3(p_3_td)
        #
        # f_3_to_2 = self.conv_3_to_2(cbam_3_td)
        # p_2_out = self.conv_2_out(
        #     self._fuse_features(
        #         weights=self.weights_2_out,
        #         features=[p_2, self.upsample(f_3_to_2)]
        #     )
        # )
        # cbam_2_out = self.cbam_3_to_2(p_2_out)
        #
        # f_2_to_3 = self.conv_2_to_3(cbam_2_out)
        # p_3_out = self.conv_3_out(
        #     self._fuse_features(
        #         weights=self.weights_3_out,
        #         features=[p_3_in, p_3_td, self.downsample(f_2_to_3)]
        #     )
        # )
        # cbam_3_out = self.cbam_2_to_3(p_3_out)
        #
        # f_3_to_4 = self.conv_3_to_4(cbam_3_out)
        # p_4_out = self.conv_4_out(
        #     self._fuse_features(
        #         weights=self.weights_4_out,
        #         features=[p_4_in, p_4_td, self.downsample(f_3_to_4)]
        #     )
        # )
        # cbam_4_out = self.cbam_3_to_4(p_4_out)
        #
        # f_4_to_5 = self.conv_4_to_5(cbam_4_out)
        # p_5_out = self.conv_5_out(
        #     self._fuse_features(
        #         weights=self.weights_5_out,
        #         features=[p_5, self.downsample(f_4_to_5)]
        #     )
        # )
        #
        # outputs = (p_2_out, p_3_out, p_4_out, p_5_out)
        # -------------------------------------------------


        # 分界线--------------------------------------------
        # dark3，dark4，dark5-------------------------------
        if self.backbone:
            out_features = self.backbone(input)  # 输入进backbone
            input = [out_features[f] for f in self.in_features]

        p_3, p_4, p_5 = input
        p_4_2 = None
        p_4_in = p_4 if p_4_2 is None else p_4_2

        f_5_to_4 = self.conv_5_to_4(p_5)
        p_4_td = self.conv_4_td(
            self._fuse_features(
                weights=self.weights_4_td,
                features=[p_4, self.upsample(f_5_to_4)]
            )
        )
        # cbam_4_td = self.cbam_5_to_4(p_4_td)


        # Out
        f_4_to_3 = self.conv_4_to_3(p_4_td)
        p_3_out = self.conv_3_out(
            self._fuse_features(
                weights=self.weights_3_out,
                features=[p_3, self.upsample(f_4_to_3)]
            )
        )
        # cbam_3_out = self.cbam_4_to_3(p_3_out)

        f_3_to_4 = self.conv_3_to_4(p_3_out)
        p_4_out = self.conv_4_out(
            self._fuse_features(
                weights=self.weights_4_out,
                features=[p_4_in, p_4_td, self.downsample(f_3_to_4)]
            )
        )
        # cbam_4_out = self.cbam_3_to_4(p_4_out)

        f_4_to_5 = self.conv_4_to_5(p_4_out)
        p_5_out = self.conv_5_out(
            self._fuse_features(
                weights=self.weights_5_out,
                features=[p_5, self.downsample(f_4_to_5)]
            )
        )

        outputs = (p_3_out, p_4_out, p_5_out)
        # -------------------------------------------------

        return outputs


    def _fuse_features(self, weights, features):
        weights = F.relu(weights)
        num = sum([w * f for w, f in zip(weights, features)]) # 做sum相加
        det = sum(weights) + self.EPS
        x = self.act(num / det)
        return x




