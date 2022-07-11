#!/usr/bin/env python

import torch.nn as nn


class FC(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 pre_fc=False,
                 p_dropout=0.0,
                 bias=True):
        """
        Single-layer Linear Neural Network

        """
        super().__init__()
        self.module_dict = nn.ModuleDict()

        self.fc = nn.Linear(input_dim, output_dim, bias=bias)
        self.module_dict['fc'] = self.fc

        self.pre_fc = None
        self.dropout = None
        if pre_fc:
            self.pre_fc = nn.Linear(input_dim, input_dim, bias=bias)
            self.module_dict['pre_fc'] = self.pre_fc
        
            if p_dropout > 0:
                self.dropout = nn.Dropout(p=p_dropout)
                self.module_dict['dropout'] = self.dropout

    def forward(self, X):
        output = X
        if self.pre_fc is not None:
            output = self.pre_fc(output)
            output = nn.ReLU()(output)

            if self.dropout is not None:
                output = self.dropout(output)

        return self.fc(output)
