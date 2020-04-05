#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.nn.functional import dropout

from mrgcn.layers.graph import GraphConvolution


class RGCN(nn.Module):
    def __init__(self, modules, num_relations, num_nodes, num_bases,
                 p_dropout, featureless, bias, link_prediction):
        """
        Relational Graph Convolutional Network

        PARAMETERS
        """
        super().__init__()

        assert len(modules) > 0

        self.num_nodes = num_nodes
        self.p_dropout = p_dropout

        self.layers = nn.ModuleDict()
        self.activations = nn.ModuleDict()
        # input layer
        indim, outdim, ltype, f_activation = modules[0]
        self.layers['layer_0'] = GraphConvolution(indim=indim,
                                          outdim=outdim,
                                          num_relations=num_relations,
                                          num_nodes=num_nodes,
                                          num_bases=num_bases,
                                          featureless=featureless,
                                          input_layer=True,
                                          bias=bias)
        self.activations['layer_0'] = f_activation

        # other layers (if any)
        for i, layer in enumerate(modules[1:], 1):
            indim, outdim, ltype, f_activation = layer
            self.layers['layer_'+str(i)] = GraphConvolution(indim=indim,
                                              outdim=outdim,
                                              num_relations=num_relations,
                                              num_nodes=num_nodes,
                                              num_bases=num_bases,
                                              featureless=False,
                                              input_layer=False,
                                              bias=bias)
            self.activations['layer_'+str(i)] = f_activation

        if link_prediction:
            # simulate diag(R) with R = (r x n x h) by vectors (r x h)
            size = (num_relations, modules[-1][1])
            self.relations = nn.Parameter(torch.empty(size))
            nn.init.xavier_uniform_(self.relations)

    def forward(self, X, A):
        # Forward pass with full batch
        for layer, f_activation in zip(self.layers.values(),
                                       self.activations.values()):
            if type(layer) is GraphConvolution:
                X = layer(X, A)
            else:
                X = layer(X)

            if self.p_dropout > 0.0:
                # add dropout to output, by elementwise multiplying with 
                # column vector of ones, with dropout applied to the vector
                # of ones.
                ones = dropout(torch.ones(self.num_nodes),
                               p=self.p_dropout)
                X = torch.mul(X.T, ones).T


            X = f_activation(X)

        return X

    def init(self):
        # reinitialze all weights
        for layer in self.layers.values():
            if type(layer) is GraphConvolution:
                layer.init()
            else:
                raise NotImplementedError
