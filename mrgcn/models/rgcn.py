#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.nn.functional import dropout

from mrgcn.data.utils import getAdjacencyNodeColumnIdx
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

            # initiate weights
            self.reset_parameters()

    def forward(self, X, A,
                batch_idx=None,
                A_neighbours_unseen=None,
                neighbours=None):

        if batch_idx is ...:
            return self._forward_full_batch(X, A)

        # X and A are slices of the whole dataset
        return self._forward_mini_batch(X, A, batch_idx,
                                        A_neighbours_unseen,
                                        neighbours)


    def _forward_full_batch(self, X, A):
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

            if f_activation is not None:
                X = f_activation(X)

        return X

    def _forward_mini_batch(self, X, A, batch_idx, A_neighbours_unseen, neighbours):
        neighbours_idx = neighbours[0]
        depth2neighbours_idx = neighbours[1]
        H_idx = neighbours[2]
        H_node_idx = neighbours[3]

        for layer, f_activation in zip(self.layers.values(),
                                       self.activations.values()):
            if layer.input_layer:
                H2 = None
                if layer.featureless:
                    H1 = layer(None, A)
                    with torch.no_grad():
                        # compute embeddings needed to compute batch nodes
                        H2 = layer(None, A_neighbours_unseen)
                else:
                    # map indices to local subset index
                    X_batch_neighbours_idx = [i for i in range(len(batch_idx))
                                              if batch_idx[i] in neighbours_idx]
                    X_batch_depth2neighbours_idx = [i for i in range(len(batch_idx))
                                                    if batch_idx[i] in depth2neighbours_idx]

                    # consider only the embeddings of connected nodes
                    A_idx = getAdjacencyNodeColumnIdx(neighbours_idx,
                                                      layer.num_nodes,
                                                      layer.num_relations)
                    H1 = layer(X[X_batch_neighbours_idx], A, A_idx)

                    if A_neighbours_unseen.shape[0] > 0:
                        # only needed if not all nodes have been computed yet
                        A_idx = getAdjacencyNodeColumnIdx(depth2neighbours_idx,
                                                          layer.num_nodes,
                                                          layer.num_relations)
                        with torch.no_grad():
                            H2 = layer(X[X_batch_depth2neighbours_idx],
                                       A_neighbours_unseen,
                                       A_idx)

                # combine embeddings necessary for next layer
                X = torch.vstack([H1, H2]) if H2 is not None else H2
            else:
                # consider only the embeddings of connected nodes
                A_idx = getAdjacencyNodeColumnIdx(H_node_idx,
                                                  layer.num_nodes,
                                                  layer.num_relations)
                X = layer(X[H_idx], A, A_idx)

            if self.p_dropout > 0.0:
                # add dropout to output, by elementwise multiplying with 
                # column vector of ones, with dropout applied to the vector
                # of ones.
                ones = dropout(torch.ones(X.shape[0]),
                               p=self.p_dropout)
                X = torch.mul(X.T, ones).T

            if f_activation is not None:
                X = f_activation(X)

        return X

    def reset_parameters(self):
        # reset the edge embeddings when doing link prediction
        nn.init.xavier_uniform_(self.relations)
