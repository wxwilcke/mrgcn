#!/usr/bin/env python

import logging
from operator import itemgetter

#import psutil
import torch
import torch.nn as nn

from mrgcn.data.utils import (collate_zero_padding,
                              scipy_sparse_list_to_pytorch_sparse)
from mrgcn.models.charcnn import CharCNN
from mrgcn.models.geomcnn import GeomCNN
from mrgcn.models.imagecnn import ImageCNN
#from mrgcn.models.rnn import RNN
from mrgcn.models.rgcn import RGCN


logger = logging.getLogger(__name__)
#PROCESS = psutil.Process()  # debug

class MRGCN(nn.Module):
    def __init__(self, modules, embedding_modules, num_relations,
                 num_nodes, num_bases=-1, p_dropout=0.0, featureless=False,
                 bias=False, link_prediction=False):
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
        self.module_dict = nn.ModuleDict()

        # add embedding layers
        self.modality_modules = dict()
        i, j, k = 0, 0, 0
        for modality, args in embedding_modules:
            if modality == "xsd.string":
                batch_size, nrows, dim_out, model_size = args
                module = CharCNN(features_in=nrows,
                                 features_out=dim_out,
                                 p_dropout=p_dropout,
                                 size=model_size)
                self.module_dict["CharCNN_"+str(i)] = module
                i += 1
            if modality == "blob.image":
                batch_size, (nchannels, nrows, ncols), dim_out = args
                module = ImageCNN(channels_in=nchannels,
                             height=nrows,
                             width=ncols,
                             features_out=dim_out,
                             p_dropout=p_dropout)
                self.module_dict["ImageCNN_"+str(j)] = module
                j += 1
            if modality == "ogc.wktLiteral":
                batch_size, nrows, dim_out = args
                module = GeomCNN(features_in=nrows,
                                 features_out=dim_out,
                                 p_dropout=p_dropout)
                self.module_dict["GeomCNN_"+str(k)] = module
                #batch_size, ncols, dim_out = args
                #module = RNN(input_dim=ncols,
                #             output_dim=dim_out,
                #             hidden_dim=ncols*2,
                #             p_dropout=p_dropout)
                #self.module_dict["RNN_"+str(k)] = module
                k += 1

            if modality not in self.modality_modules.keys():
                self.modality_modules[modality] = list()
            self.modality_modules[modality].append((module, batch_size))

        # add graph convolution layers
        self.rgcn = RGCN(modules, num_relations, num_nodes,
                          num_bases, p_dropout, featureless, bias,
                          link_prediction)
        self.module_dict["RGCN"] = self.rgcn

    def forward(self, X, A, batch_grad_idx=-1, device=None):
        X, F = X[0], X[1:]

        # compute and concat modality-specific embeddings
        XF = self._compute_modality_embeddings(F,
                                               batch_grad_idx,
                                               device)
        if XF is not None:
            #logger.debug(" Merging structure and node features")
            X = torch.cat([X,XF], dim=1)

        # Forward pass through graph convolution layers
        #logger.debug(" Forward pass with input of size {} x {}".format(X.size(0),
        #                        X.size(1)))
        self.rgcn.to(device)
        X_dev = X.to(device)
        A_dev = A.to(device)

        X_dev = self.rgcn(X_dev, A_dev)
        X = X_dev.to('cpu')

        return X

    def _compute_modality_embeddings(self, F, batch_grad_idx, device):
        X = list()
        for modality, F_set in F:
            if modality not in self.modality_modules.keys() or len(F_set[0]) <= 0:
                continue

            for i, ((encodings, node_idx, C, _, nsets), batches) in enumerate(F_set):
                module, _ = self.modality_modules[modality][i]
                module.to(device)

                out = list()
                out_node_idx = list()
                nbatches = len(batches)
                for j, (batch_encoding_idx, batch_node_idx) in enumerate(batches):
                    if len(batch_encoding_idx) <= 0:
                        # dirty bug fix
                        continue

                    if modality in ["xsd.string", "ogc.wktLiteral"]:
                        # encodings := list of sparse coo matrices
                        batch = itemgetter(*batch_encoding_idx)(encodings)
                        if type(batch) is not tuple:  # single sample
                            batch = (batch,)

                        time_dim = 1 # if modality == "xsd.string" else 0  ## uncomment for RNN
                        batch = collate_zero_padding(batch,
                                                     time_dim)

                        batch = scipy_sparse_list_to_pytorch_sparse(batch)
                        batch = batch.to_dense()
                    else:
                        # encodings := numpy array
                        batch = encodings[batch_encoding_idx]
                        batch = torch.as_tensor(batch)

                    # forward pass
                    #print(" {} - {} GB".format(batch.size(),
                    #                          PROCESS.memory_info().rss/1e9))
                    batch_dev = batch.to(device)
                    if batch_grad_idx < 0:
                        # compute gradients on whole dataset
                        logger.debug(" {} (set {} / {}) - batch {} / {} +grad".format(modality,
                                                                           i+1, nsets,
                                                                           j+1, nbatches))
                        out_dev = module(batch_dev)
                    else:
                        # compute gradients on one batch per epoch
                        if batch_grad_idx % nbatches == j:
                            logger.debug(" {} (set {} / {}) - batch {} / {} +grad".format(modality,
                                                                               i+1, nsets,
                                                                               j+1, nbatches))
                            out_dev = module(batch_dev)
                        else:
                            with torch.no_grad():
                                logger.debug(" {} (set {} / {}) - batch {} / {} -grad".format(modality,
                                                                                   i+1, nsets,
                                                                                   j+1, nbatches))
                                out_dev = module(batch_dev)

                    out_cpu = out_dev.to('cpu')
                    out.append(out_cpu)
                    out_node_idx.extend(batch_node_idx)

                out = torch.cat(out, dim=0)

                # map output to correct nodes
                XF = torch.zeros((self.num_nodes, C), dtype=torch.float32)
                XF[out_node_idx] = out

                X.append(XF)

        return None if len(X) <= 0 else torch.cat(X, dim=1)

    def init(self):
        # reinitialze all weights
        for module in self.module_dict.values():
            #if type(module) in (ImageCNN, CharCNN, RGCN, RNN):
            if type(module) in (ImageCNN, CharCNN, GeomCNN, RGCN):
                module.init()
            else:
                raise NotImplementedError
