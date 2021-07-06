#!/usr/bin/env python

import torch.nn as nn


class CharCNN(nn.Module):
    def __init__(self, features_in, features_out, p_dropout=0.0, size="M"):
        """
        Character-level Convolutional Neural Network

        features_in  :: size of alphabet (nrows of input matrix, default 37)
        features_out :: size of final layer
        size         :: 'S' small, 'M' medium, or 'L' large network

        """
        super().__init__()

        if size == "S":
            self.minimal_length = 5
            # sequence length >= 5
            self.conv = nn.Sequential(
                nn.Conv1d(features_in, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),

                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),  # len == 1
                nn.ReLU()
            )

            n_fc = max(128, features_out)
            self.fc = nn.Sequential(
                nn.Linear(128, n_fc),
                nn.ReLU(),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_fc, features_out)
            )
        elif size == "M":
            self.minimal_length = 20
            # sequence length >= 20
            self.conv = nn.Sequential(
                nn.Conv1d(features_in, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),

                nn.Conv1d(64, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),

                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),

                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 512, kernel_size=3, padding=1),  # len == 1
                nn.ReLU()
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
        elif size == "L":
            self.minimal_length = 100
            # sequence length >= 100
            self.conv = nn.Sequential(
                nn.Conv1d(features_in, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),

                nn.Conv1d(64, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),

                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(5),

                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 512, kernel_size=3),
                nn.ReLU(),
                nn.Conv1d(512, 1024, kernel_size=3),  # len == 1
                nn.ReLU()
            )

            n_first = max(512, features_out)
            n_second = max(128, features_out)
            self.fc = nn.Sequential(
                nn.Linear(1024, n_first),
                nn.ReLU(),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_first, n_second),
                nn.ReLU(),
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
