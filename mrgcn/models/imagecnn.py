#!/usr/bin/env python

from functools import partial

import torch.nn as nn
from torchvision.models.mobilenetv3 import (InvertedResidualConfig,
                                            MobileNetV3)


class ImageCNN(nn.Module):
    def __init__(self, features_out=1000, p_dropout=0.2,
                 bias=True):
        super().__init__()

        inverted_residual_setting, last_channel = self.conf()
        self.model = MobileNetV3(inverted_residual_setting, last_channel)

        # change first layer to prevent drop in dimension (out = 64^2, 16).
        self.model._modules['features'][0][0] = nn.Conv2d(3, 16,
                                                          kernel_size=(3, 3),
                                                          stride=(1, 1))

        # no need to change last conv layer since implementation uses
        # adaptive pool operator

        # change dropout
        dropout = self.model._modules['classifier'][-2]
        if dropout.p != p_dropout:
            self.model._modules['classifier'][-2] = nn.Dropout(p=p_dropout)

        # change output features
        classifier = self.model._modules['classifier'][-1]
        if features_out != classifier.out_features:
            features_in = classifier.in_features
            fc = nn.Linear(in_features=features_in,
                           out_features=features_out,
                           bias=bias)

            self.model._modules['classifier'][-1] = fc

    def conf(self):
        reduce_divider = 1
        dilation = 1
        width_mult = 1.0

        bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_mult=width_mult)

        inverted_residual_setting = [
            # second bneck_conf differs from paper
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # out 32,16
            bneck_conf(16, 3, 72, 24, False, "RE", 1, 1),  # out 32,24 ++
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),  # out 32,24
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # out 16,40
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),  # out 16,40
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),  # out 16,40
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),  # out 16,48
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),  # out 16,48
            bneck_conf(48, 5, 288, 96 // reduce_divider,
                       True, "HS", 2, dilation),  # out 8,96
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider,
                       96 // reduce_divider, True, "HS", 1, dilation),  # 8,96
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider,
                       96 // reduce_divider, True, "HS", 1, dilation),  # 8,96
        ]
        last_channel = adjust_channels(1024 // reduce_divider)

        return (inverted_residual_setting, last_channel)

    def forward(self, X):
        return self.model(X)
