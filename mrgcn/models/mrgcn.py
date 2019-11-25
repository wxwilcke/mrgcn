#!/usr/bin/env python

import logging

import torch
import torch.nn as nn
import torch.utils.data as td

from mrgcn.models.charcnn import CharCNN
from mrgcn.models.rgcn import RGCN


logger = logging.getLogger(__name__)

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
        self.module_list = nn.ModuleList()

        # add embedding layers
        self.modality_modules = dict()
        for modality, args in embedding_modules:
            if modality == "xsd.string":
                batch_size, (nrows, ncols), dim_out = args
                module = CharCNN(features_in=nrows,
                                 features_out=dim_out,
                                 sequence_length=ncols,
                                 p_dropout=p_dropout)
                self.module_list.append(module)
            if modality == "blob.image":
                pass

            if modality not in self.modality_modules.keys():
                self.modality_modules[modality] = list()
            self.modality_modules[modality].append((module, batch_size))

        # add graph convolution layers
        self.mrgcn = RGCN(modules, num_relations, num_nodes,
                          num_bases, p_dropout, featureless, bias)
        self.module_list.append(self.mrgcn)

    def forward(self, X, A, device=None):
        X, F_string, F_images = X

        # compute and concat modality-specific embeddings
        XF = self._compute_modality_embeddings(F_string,
                                               F_images,
                                               device)
        if XF is not None:
            X = torch.cat([X,XF], dim=1)

        # Forward pass through graph convolution layers
        self.mrgcn.to(device)
        X_dev = X.to(device)
        A_dev = A.to(device)

        X_dev = self.mrgcn(X_dev, A_dev)
        X = X_dev.to('cpu')

        return X

    def _compute_modality_embeddings(self, F_string, F_images, device):
        X = list()
        for modality, F in zip(["xsd.string", "blob.image"],
                               [F_string, F_images]):
            if modality not in self.modality_modules.keys() or F is None:
                continue

            #logging.debug("Computing modality specific embeddings for {}".format(modality))
            for i, (encodings, node_idx, C, _) in enumerate(F):
                module, batch_size = self.modality_modules[modality][i]
                module.to(device)

                encodings = torch.as_tensor(encodings)  # convert from numpy array
                data_loader = td.DataLoader(td.TensorDataset(encodings),
                                            batch_size=batch_size,
                                            shuffle=False)  # order matters
                out = list()
                for [batch] in data_loader:
                    batch_dev = batch.to(device)
                    out_dev = module(batch_dev)

                    out_cpu = out_dev.to('cpu')
                    out.append(out_cpu)

                out = torch.cat(out, dim=0)

                # map output to correct nodes
                XF = torch.zeros((self.num_nodes, C), dtype=torch.float32)
                XF[node_idx] = out

                X.append(XF)

        return None if len(X) <= 0 else torch.cat(X, dim=1)

    def init(self):
        # reinitialze all weights
        for module in self.module_list:
            if type(module) in (CharCNN, RGCN):
                module.init()
            else:
                raise NotImplementedError
