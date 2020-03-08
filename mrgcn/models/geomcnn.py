#!/usr/bin/env python

import torch.nn as nn


class GeomCNN(nn.Module):
    def __init__(self, features_in, features_out, p_dropout=0.0):
        """
        Temporal Convolutional Neural Network to learn geometries

        features_in  :: size of point encoding (nrows of input matrix)
        features_out :: size of final layer

        Based on architecture described in:

            van't Veer, Rein, Peter Bloem, and Erwin Folmer.
            "Deep Learning for Classification Tasks on Geospatial
            Vector Polygons." arXiv preprint arXiv:1806.03857 (2018).
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(features_in, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=3),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8)  # out = 8 x 64 = 512
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),

            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),

            nn.Linear(32, features_out)
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
