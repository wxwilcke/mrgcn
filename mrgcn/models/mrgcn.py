#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import dropout

from mrgcn.layers.graph import GraphConvolution


class MRGCN(nn.Module):
    def __init__(self, modules, embedding_modules, num_relations,
                 num_nodes, num_bases=-1, p_dropout=0.0, featureless=False,
                 bias=False):
        """
        Multimodal Relational Graph Convolutional Network

        PARAMETERS
            modules:    list with tuples (input dimension,
                                          output dimension,
                                          layer type,
                                          activation function)
        """
        super().__init__()

        assert len(modules) > 0

        self.num_nodes = num_nodes
        self.p_dropout = p_dropout

        # add additional embedding layers
        self.embedding_layers = nn.ModuleDict()
        for modality, embedding_module in embedding_modules:
            self.embedding_layers[modality](self.PreEmbedding(embedding_module))
            # add to parameters?

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

    def forward(self, X, A):
        X, F_string, F_images = X

        # compute and concat modality-specific embeddings
        XF = self.compute_modality_embeddings(F_string,
                                              F_images)
        if XF is not None:
            X = torch.cat([X,XF], dim=1)

        # Forward pass with full input
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

    def compute_modality_embeddings(self, F_string, F_images):
        X = list()
        for modality, F in zip(["xsd.string", "blob.image"],
                               [F_string, F_images]):
            if F is None:
                continue

            encodings, node_idx, C, _ = F
            XF = np.zeros((self.num_nodes, C), dtype=np.float32)
            for i, x in enumerate(encodings):
                # do we already need to have pytorch tensors here?
                XF[node_idx[i]] = self.embedding_layers[modality](x)

        return None if len(X) <= 0 else torch.as_tensor(np.hstack(X))

    def reset(self):
        # reinitialze all weights
        for layer in self.layers.values():
            if type(layer) is GraphConvolution:
                layer.init()
            else:
                raise NotImplementedError

        for layer in self.embedding_layers.values():
            layer.init()

class PreEmbedding(nn.Module):
    def __init__(self, modules):
        """
        PARAMETERS
            modules:    list with tuples (module, activation function)
        """
        super().__init__()

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for module, f_activation in modules:
            self.layers.append(module)
            self.activations.append(f_activation)

    def forward(self, X):
        for layer, f_activation in zip(self.layers,
                                       self.activations):
            X = f_activation(layer(X))

        return X

    def init(self):
        # reinitialze all weights
        for layer in self.layers:
            for param in layer.parameters():
                # initialize weights from a uniform distribution following 
                # "Understanding the difficulty of training deep feedforward 
                #  neural networks" - Glorot, X. & Bengio, Y. (2010)
                nn.init.xavier_uniform_(param)
