#!/usr/bin/env python

import torch.nn as nn


class CharCNN(nn.Module):
    def __init__(self, features_in, features_out, p_dropout=0.0, size="M"):
        """
        Character-level Convolutional Neural Network

        features_in  :: size of alphabet (nrows of input matrix, default 37)
        features_out :: size of final layer
        size         :: 'S' small, 'M' medium, or 'L' large network

        Based on architecture described in:

            Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional
            Networks for Text Classification. Advances in Neural Information
            Processing Systems 28 (NIPS 2015)
        """
        super().__init__()

        if size == "S":
            # sequence length >= 3
            self.conv = nn.Sequential(
                nn.Conv1d(features_in, 64, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=3),  # len/3

                nn.Conv1d(64, 64, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool1d(4)
            )

            n_fc = max(32, features_out)
            self.fc = nn.Sequential(
                nn.Linear(256, n_fc),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_fc, features_out)
            )
        elif size == "M":
            # sequence length >= 12
            self.conv = nn.Sequential(
                nn.Conv1d(features_in, 64, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),  # len/2

                nn.Conv1d(64, 64, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),  # len/4

                nn.Conv1d(64, 64, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, kernel_size=7, padding=2),  # (len/4) - 2
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool1d(8)
            )

            n_first = max(256, features_out)
            n_second = max(64, features_out)
            self.fc = nn.Sequential(
                nn.Linear(512, n_first),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_first, n_second),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_second, features_out)
            )
        elif size == "L":
            # sequence length >= 30
            self.conv = nn.Sequential(
                nn.Conv1d(features_in, 64, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=3),  # len/3

                nn.Conv1d(64, 128, kernel_size=8),  # len/3 - 7
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=3),  # (len/3 - 7)/3

                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool1d(8)
            )

            n_first = max(512, features_out)
            n_second = max(128, features_out)
            self.fc = nn.Sequential(
                nn.Linear(1024, n_first),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_first, n_second),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_second, features_out)
            )

    def forward(self, X):
        X = self.conv(X)
        X = X.view(X.size(0), -1)
        X = self.fc(X)

        return X

    def init(self):
        for param in self.parameters():
            nn.init.normal_(param)

def out_dim(seq_length, kernel_size, padding=0, stride=1, dilation=1):
    return (seq_length+2*padding-dilation*(kernel_size-1)-1)//stride+1
