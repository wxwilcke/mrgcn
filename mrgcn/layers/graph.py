#!/usr/bin/env python

import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    def __init__(self, indim, outdim, num_relations, num_nodes, num_bases=-1,
                 bias=False, input_layer=False, featureless=False,
                 shared_bases_weights=False):
        """
        Relational Graph Convolutional Layer
        """
        super().__init__()

        self.indim = indim  # no. of input features
        self.outdim = outdim  # no. of output features
        self.num_relations = num_relations  # no. of relations
        self.num_nodes = num_nodes  # no. of nodes (batch size)
        self.num_bases = num_bases
        self.input_layer = input_layer
        self.featureless = featureless
        self.bias = bias

        self.W_I = None
        self.W_F = None
        self.W_I_comp = None
        self.W_F_comp = None
        self.b = None

        S = self.num_relations
        if self.num_bases > 0:
            # weights for bases matrices for identities and features
            S = self.num_bases

            if self.input_layer:
                self.W_I_comp = nn.Parameter(torch.empty((self.num_relations,
                                                          self.num_bases)))
            if not self.featureless:
                if shared_bases_weights:
                    # use same basis matrix for both identities and features
                    self.W_F_comp = self.W_I_comp
                else:
                    self.W_F_comp = nn.Parameter(torch.empty((self.num_relations,
                                                              self.num_bases)))

        # weights for identities and features
        if self.input_layer:
            self.W_I = nn.Parameter(torch.empty((S*self.num_nodes, self.outdim)))
        if not self.featureless:
            self.W_F = nn.Parameter(torch.empty((S, self.indim, self.outdim)))

        # declare bias
        if self.bias:
            self.b = nn.Parameter(torch.empty(self.outdim))

        # initialize weights
        self.init()

    def forward(self, X, A):
        # if input layer: AXW = A[I F]W = AIW_I + AFW_F
        # else:           AXW = AHW

        AIW_I = 0.0
        if self.input_layer:
            W_I = self.W_I
            if self.num_bases > 0:
                W_I = W_I.view(self.num_bases, self.num_nodes, self.outdim)
                W_I = torch.einsum('rb,bij->rij', self.W_I_comp, W_I)
                W_I = W_I.view(self.num_relations*self.num_nodes, self.outdim)

            # AIW_I = AW_I
            AIW_I = torch.mm(A, W_I)

            if self.featureless:
                if self.bias:
                    AIW_I = torch.add(AIW_I, self.b)

                return AIW_I

        W_F = self.W_F
        if self.num_bases > 0:
            W_F = torch.einsum('rb,bij->rij', self.W_F_comp, W_F)

        FW_F = torch.einsum('ij,bjk->bik', X, W_F)
        FW_F = torch.reshape(FW_F, (self.num_relations*self.num_nodes, self.outdim))
        AFW_F = torch.mm(A, FW_F)

        AXW = torch.add(AIW_I, AFW_F) if self.input_layer else AFW_F

        if self.bias:
            AXW = torch.add(AXW, self.b)

        return AXW

    def init(self):
        # initialize weights from a uniform distribution following 
        # "Understanding the difficulty of training deep feedforward 
        #  neural networks" - Glorot, X. & Bengio, Y. (2010)
        for name, param in self.named_parameters():
            if name == 'b':
                # skip bias
                continue
            nn.init.xavier_uniform_(param)

        # initialize bias as null vector
        if self.bias:
            nn.init.zeros_(self.b)
