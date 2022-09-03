#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from torch import nn
import torch
from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck, Transpose
from .cbam import CBAM
from .non_local import NonLocalBlock
from .ccnet import CrissCrossAttention
from .ca import CABlock


class Tranpose_CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        # out_features=("dark2", "dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
            # CBAM(base_channels * 2, base_channels * 2),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
            # CrissCrossAttention(base_channels * 4),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
            # CABlock(base_channels * 8, base_channels * 8),
            # CrissCrossAttention(base_channels * 8),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
            # CABlock(base_channels * 16, base_channels * 16),
            # CrissCrossAttention(base_channels * 16),
        )

        # self.out3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        # self.out4 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        # self.out5 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.out3 = Transpose(256, 128, 4, 2, 1)
        self.out4 = Transpose(512, 256, 4, 2, 1)
        self.out5 = Transpose(1024, 512, 4, 2, 1)



    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x

        x = self.dark2(x)
        y = x
        outputs["dark2"] = x

        x = self.dark3(x)
        z = self.out3(x)
        z = torch.cat([y, z], 1)
        y = x
        outputs["dark3"] = z

        x = self.dark4(x)
        z = self.out4(x)
        z = torch.cat([y, z], 1)
        y = x
        outputs["dark4"] = z

        x = self.dark5(x)
        z = self.out5(x)
        z = torch.cat([y, z], 1)
        outputs["dark5"] = z
        return {k: v for k, v in outputs.items() if k in self.out_features}
