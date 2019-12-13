#!/usr/bin/env python

from math import sqrt

import torch
import torch.nn as nn


#class RNN(nn.Module):
#    def __init__(self,
#                 input_dim,
#                 output_dim,
#                 hidden_dim,
#                 sequence_length,
#                 p_dropout=0.0):
#        """
#        Recurrent Neural Network
#
#        """
#        super().__init__()
#        self.hidden_dim = hidden_dim
#
#        self.fc_loop = nn.Linear(input_dim + hidden_dim, hidden_dim)
#        self.fc_out = nn.Linear(input_dim + hidden_dim, output_dim)
#        self.f_activation = nn.ReLU()
#
#    def forward(self, X, H=None):
#        if H is None:
#            H = torch.zeros(self.hidden_dim)
#
#        XH = torch.cat([X, H], dim=1)
#        H  = self.fc_loop(XH)
#        XH = self.fc_out(XH)
#
#        return (self.f_activation(X), H)
#
#    def init(self):
#        for param in self.parameters():
#            nn.init.normal_(param)

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

    def init(self):
        sqrt_k = sqrt(1.0/self.hidden_dim)
        for param in self.parameters():
            nn.init.uniform_(param, -sqrt_k, sqrt_k)
