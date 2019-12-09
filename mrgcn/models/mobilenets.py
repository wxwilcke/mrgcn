#!/usr/bin/env python

import torch.nn as nn


class MobileNETS(nn.Module):
    def __init__(self, channels_in, height, width, features_out=1000, p_dropout=0.0):
        """
        Image Convolutional Neural Network

        Implementation based on work from:

            Howard, Andrew G., et al. "Mobilenets: Efficient convolutional
            neural networks for mobile vision applications." arXiv preprint
            arXiv:1704.04861 (2017).

        """
        super().__init__()

        def conv_std(channels_in, channels_out, stride):
            # standard convolutional layer
            return nn.Sequential(
                                nn.Conv2d(channels_in, channels_out,
                                          kernel=(3, 3),
                                          stride=stride,
                                          padding=1),
                                nn.BatchNorm2d(channels_out),
                                nn.ReLU(inplace=True)
            )

        def conv_ds(channels_in, channels_out, stride):
            # depthwise separable convolutions
            return nn.Sequential(
                                conv_dw(channels_in, channels_in, stride),
                                conv_pw(channels_in, channels_out, stride)
            )

        def conv_dw(channels_in, channels_out, stride):
            # depthwise convolutional layer
            return nn.Sequential(
                                nn.Conv2d(channels_in, channels_out,
                                          kernel=(3, 3),
                                          stride=stride,
                                          padding=1,
                                          groups=channels_in),
                                nn.BatchNorm2d(channels_out),
                                nn.ReLU(inplace=True),
            )

        def conv_pw(channels_in, channels_out, stride):
            # pointwise convolutional layer
            return nn.Sequential(
                                nn.Conv2d(channels_in, channels_out,
                                          kernel=(1, 1),
                                          stride=1,
                                          padding=0),
                                nn.BatchNorm2d(channels_out),
                                nn.ReLU(inplace=True)
            )

        self.conv = nn.Sequential(
            conv_std(  3,   32, 2),
            conv_ds(  32,   64, 1),
            conv_ds(  64,  128, 2),
            conv_ds( 128,  128, 1),
            conv_ds( 128,  256, 2),
            conv_ds( 256,  256, 1),
            conv_ds( 256,  512, 2),
            conv_ds( 512,  512, 1),
            conv_ds( 512,  512, 1),
            conv_ds( 512,  512, 1),
            conv_ds( 512,  512, 1),
            conv_ds( 512,  512, 1),
            conv_ds( 512, 1024, 2),
            conv_ds(1024, 1024, 1),  # wrong stride value, 2, in paper
            nn.AvgPool2d(7, stride=1)
        )
        self.fc = nn.Linear(1024, features_out)

    def forward(self, X):
        X = self.conv(X)
        X = X.view(-1, 1024)

        return self.fc(X)

    def init(self):
        # initialize weights from a uniform distribution following 
        # "Understanding the difficulty of training deep feedforward 
        #  neural networks" - Glorot, X. & Bengio, Y. (2010)
        for param in self.parameters():
            nn.init.xavier_uniform_(param)
