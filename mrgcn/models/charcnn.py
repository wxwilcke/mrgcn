#!/usr/bin/env python

import torch.nn as nn


class CharCNN(nn.Module):
    def __init__(self, features_in, features_out, p_dropout=0.0):
        """
        Character-level Convolutional Neural Network


        Based on architecture described in:

            Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional
            Networks for Text Classification. Advances in Neural Information
            Processing Systems 28 (NIPS 2015)
        """
        super().__init__()

        def conv1d(features_in, features_out, kernel, padding=0):
            return nn.Sequential(
                nn.Conv1d(features_in, features_out,
                          kernel_size=kernel,
                          padding=padding,
                          stride=1),
                nn.ReLU(inplace=True)
            )

        def linear(features_in, features_out):
            return nn.Sequential(
                nn.Linear(features_in, features_out),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_dropout)
            )

        self.conv = nn.Sequential(
            conv1d(features_in, 256, kernel=7, padding=3),  # account for short sequences
            nn.AdaptiveMaxPool1d(336),   # in variable
            conv1d(256, 256, kernel=7),  # in 336
            nn.MaxPool1d(kernel_size=3, stride=3),  # in 330
            conv1d(256, 256, kernel=3),  # in 110
            conv1d(256, 256, kernel=3),  # in 108
            conv1d(256, 256, kernel=3),  # in 106
            conv1d(256, 256, kernel=3),  # in 104
            nn.MaxPool1d(kernel_size=3, stride=3)  # in 102, out 34
        )

        self.fc = nn.Sequential(
            linear(8704, 1024),
            linear(1024, 1024),
            linear(1024, features_out)
        )

    def forward(self, X):
        X = self.conv(X)
        X = X.view(X.size(0), -1)  # B x 256 x 34 -> B x 8704
        X = self.fc(X)

        return X

    def init(self):
        for param in self.parameters():
            nn.init.normal_(param)

def out_dim(seq_length, kernel_size, padding=0, stride=1, dilation=1):
    return (seq_length+2*padding-dilation*(kernel_size-1)-1)//stride+1
