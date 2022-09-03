#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import math
from .darknet import CSPDarknet
from .double_darknet import Double_CSPDarknet
from .new_double_darknet import New_Double_CSPDarknet
from .swin_transformer import swin_base_patch4_window7_224
from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOSWIN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()

        # self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        # self.backbone = swin_base_patch4_window7_224()


        self.backbone = swin_base_patch4_window7_224()
        self.embed_dim = swin_base_patch4_window7_224().embed_dim

        # self.feature32x2feat3 = nn.Conv2d(1024, int(in_channels[2] * width), kernel_size=1)
        # self.feature16x2feat2 = nn.Conv2d(512, int(in_channels[1] * width), kernel_size=1)
        # self.feature8x2feat1 = nn.Conv2d(256, int(in_channels[0] * width), kernel_size=1)

        self.feature32x2feat3 = nn.Conv2d(self.embed_dim * 8, int(in_channels[2] * width), kernel_size=1)
        self.feature16x2feat2 = nn.Conv2d(self.embed_dim * 4, int(in_channels[1] * width), kernel_size=1)
        self.feature8x2feat1 = nn.Conv2d(self.embed_dim * 2, int(in_channels[0] * width), kernel_size=1)

        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        # out_features = self.backbone(input)
        # features = [out_features[f] for f in self.in_features]
        # [x2, x1, x0] = features

        feature4x, feature8x, feature16x, feature32x, feature64x = self.backbone.forward(input)
        # feature4x, feature8x, feature16x, feature32x, feature64x = self.backbone(input)
        # print("orignalfeature32size:", feature32x.size())
        # print("orignalfeature16size:", feature16x.size())
        # print("orignalfeature8size:", feature8x.size())
        feature32x_sqrt = int(math.sqrt(feature32x.size()[1]))
        feature16x_sqrt = int(math.sqrt(feature16x.size()[1]))
        feature8x_sqrt = int(math.sqrt(feature8x.size()[1]))

        channel_feature32 = feature32x.size()[2]
        channel_feature16 = feature16x.size()[2]
        channel_feature8 = feature8x.size()[2]

        feature32x = feature32x.permute(0, 2, 1).contiguous().view(-1, channel_feature32, feature32x_sqrt, feature32x_sqrt)
        # print("after reshape feature32:", feature32x.size())
        feature16x = feature16x.permute(0, 2, 1).contiguous().view(-1, channel_feature16, feature16x_sqrt, feature16x_sqrt)
        # print("after reshape feature16:", feature16x.size())
        feature8x = feature8x.permute(0, 2, 1).contiguous().view(-1, channel_feature8, feature8x_sqrt, feature8x_sqrt)
        # print("after reshpae feature8:", feature8x.size())

        # feature32x = feature32x.permute(0, 2, 1).contiguous().view(-1, 1024, feature32x_sqrt, feature32x_sqrt)
        # # print("after reshape feature32:", feature32x.size())
        # feature16x = feature16x.permute(0, 2, 1).contiguous().view(-1, 512, feature16x_sqrt, feature16x_sqrt)
        # # print("after reshape feature16:", feature16x.size())
        # feature8x = feature8x.permute(0, 2, 1).contiguous().view(-1, 256, feature8x_sqrt, feature8x_sqrt)
        # # print("after reshpae feature8:", feature8x.size())

        x0 = self.feature32x2feat3(feature32x)
        x1 = self.feature16x2feat2(feature16x)
        x2 = self.feature8x2feat1(feature8x)

        # x0 = feat3
        # x1 = feat2
        # x2 = feat1

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        # outputs = (pan_out2, pan_out1)
        return outputs
