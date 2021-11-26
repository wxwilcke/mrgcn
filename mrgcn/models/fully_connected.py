#!/usr/bin/env python

import torch.nn as nn
import torch.nn.functional as f


class FC(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 p_dropout=0.0):
        """
        Single-layer Linear Neural Network

        """
        super().__init__()

        self.p_dropout = p_dropout
        self.fc = nn.Linear(input_dim, output_dim)

        # initiate weights
        self.init()

    def forward(self, X):
        X = self.fc(X)

        return f.dropout(X, p=self.p_dropout)

    def init(self):
        for param in self.parameters():
            nn.init.uniform_(param)
