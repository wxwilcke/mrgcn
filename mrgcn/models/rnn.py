#!/usr/bin/env python

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 p_dropout=0.0):
        """
        Recurrent Neural Network

        """
        super().__init__()
        self.hidden_dim = hidden_dim

        self.rnn = nn.RNN(input_size=input_dim,
                          hidden_size=hidden_dim,
                          nonlinearity='relu',
                          bias=True,
                          batch_first=True,  # (batch, seq, feature)
                          dropout=p_dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        # default H0 is zero vector
        # output Hn is representation of entire sequence
        _, H = self.rnn(X)
        X = torch.squeeze(H, dim=0)

        return self.fc(X)
