#!/usr/bin/env python

import torch.nn as nn


class GeomCNN(nn.Module):
    def __init__(self, features_in, features_out, p_dropout=0.0):
        """
        Temporal Convolutional Neural Network to learn geometries

        features_in  :: size of point encoding (default 9)
        features_out :: size of final layer

        Minimal sequence length = 12
        """
        super().__init__()

        self.minimal_length = 12
        self.conv = nn.Sequential(
            nn.Conv1d(features_in, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(3),

            nn.Conv1d(256, 512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        n_first = max(256, features_out)
        n_second = max(128, features_out)
        self.fc = nn.Sequential(
            nn.Linear(512, n_first),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),

            nn.Linear(n_first, n_second),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),

            nn.Linear(n_second, features_out)
        )

        # initiate weights
        self.init()

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
