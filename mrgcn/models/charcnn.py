#!/usr/bin/env python

import torch.nn as nn


class CharCNN(nn.Module):
    def __init__(self, features_in, features_out, sequence_length, p_dropout=0.0):
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
            conv1d(features_in, 256, kernel=7, padding=3),
            nn.AdaptiveMaxPool1d(16),
            conv1d(256, 256, kernel=7),
            nn.AdaptiveMaxPool1d(10),
            conv1d(256, 256, kernel=3),
            conv1d(256, 256, kernel=3),
            conv1d(256, 256, kernel=3),
            conv1d(256, 256, kernel=3),
            nn.AdaptiveMaxPool1d(4)  # 1024/256
        )

        self.fc = nn.Sequential(
            linear(1024, 512),
            linear(512, features_out)
        )

    def forward(self, X):
        X = self.conv(X)
        X = X.view(X.size(0), -1)
        X = self.fc(X)

        return X

    def init(self):
        # initialize weights from a uniform distribution following 
        # "Understanding the difficulty of training deep feedforward 
        #  neural networks" - Glorot, X. & Bengio, Y. (2010)
        for param in self.parameters():
            nn.init.xavier_uniform_(param)
