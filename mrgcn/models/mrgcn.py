#!/usr/bin/env python

import logging
from operator import itemgetter

import torch
import torch.nn as nn

from mrgcn.data.utils import (collate_zero_padding,
                              scipy_sparse_list_to_pytorch_sparse)
from mrgcn.models.fully_connected import FC
from mrgcn.models.imagecnn import ImageCNN
#from mrgcn.models.rnn import RNN
from mrgcn.models.rgcn import RGCN
from mrgcn.models.temporal_cnn import TCNN


logger = logging.getLogger(__name__)


class MRGCN(nn.Module):
    def __init__(self, modules, embedding_modules, num_relations,
                 num_nodes, num_bases=-1, p_dropout=0.0, featureless=False,
                 bias=False, link_prediction=False):
        """
        Multimodal Relational Graph Convolutional Network

        """
        super().__init__()

        assert len(modules) > 0

        self.num_nodes = num_nodes
        self.p_dropout = p_dropout
        self.module_dict = nn.ModuleDict()

        # add embedding layers
        self.modality_modules = dict()
        self.modality_out_dim = 0
        self.compute_modality_embeddings = False
        h, i, j, k = 0, 0, 0, 0
        for datatype, args in embedding_modules:
            seq_length = -1

            if datatype in ["xsd.boolean", "xsd.numeric"]:
                ncols, dim_out, dropout = args
                module = FC(input_dim=ncols,
                            output_dim=dim_out,
                            p_dropout=dropout)
                self.module_dict["FC_num_"+str(i)] = module
                h += 1
            if datatype in ["xsd.date", "xsd.dateTime", "xsd.gYear"]:
                ncols, dim_out, dropout = args
                module = FC(input_dim=ncols,
                            output_dim=dim_out,
                            p_dropout=dropout)
                self.module_dict["FC_temp_"+str(i)] = module
                h += 1
            if datatype in ["xsd.string", "xsd.anyURI"]:
                nrows, dim_out, model_size, dropout = args
                module = TCNN(features_in=nrows,
                              features_out=dim_out,
                              p_dropout=dropout,
                              size=model_size)
                self.module_dict["CharCNN_"+str(i)] = module
                seq_length = module.minimal_length
                i += 1
            if datatype == "blob.image":
                (nchannels, nrows, ncols), dim_out, dropout = args
                module = ImageCNN(features_out=dim_out,
                                  p_dropout=dropout)
                #module = ImageCNN(channels_in=nchannels,
                #             height=nrows,
                #             width=ncols,
                #             features_out=dim_out,
                #             p_dropout=p_dropout)
                self.module_dict["ImageCNN_"+str(j)] = module
                j += 1
            if datatype == "ogc.wktLiteral":
                nrows, dim_out, model_size, dropout = args
                module = TCNN(features_in=nrows,
                              features_out=dim_out,
                              p_dropout=dropout,
                              size=model_size)
                seq_length = module.minimal_length
                self.module_dict["GeomCNN_"+str(k)] = module
                #ncols, dim_out = args
                #module = RNN(input_dim=ncols,
                #             output_dim=dim_out,
                #             hidden_dim=ncols*2,
                #             p_dropout=p_dropout)
                #self.module_dict["RNN_"+str(k)] = module
                k += 1

            if datatype not in self.modality_modules.keys():
                self.modality_modules[datatype] = list()
            self.modality_modules[datatype].append((module, seq_length, dim_out))
            self.modality_out_dim += dim_out
            self.compute_modality_embeddings = True

            if seq_length > 0:
                logger.debug(f"Setting sequence length to {seq_length} for datatype {datatype}")

        # add graph convolution layers
        self.rgcn = RGCN(modules, num_relations, num_nodes,
                          num_bases, p_dropout, featureless, bias,
                          link_prediction)
        self.module_dict["RGCN"] = self.rgcn

    def forward(self, X, A, epoch, device=None):
        X, F = X[0], X[1:]

        # compute and concat modality-specific embeddings
        if self.compute_modality_embeddings:
            XF = self._compute_modality_embeddings(F, epoch,
                                                   device)

            #logger.debug(" Merging structure and node features")
            X = torch.cat([X,XF], dim=1)

        # Forward pass through graph convolution layers
        #logger.debug(" Forward pass with input of size {} x {}".format(X.size(0),
        #                        X.size(1)))
        self.rgcn.to(device)
        X_dev = X.to(device)
        A_dev = A.to(device)

        X_dev = self.rgcn(X_dev, A_dev)

        return X_dev

    def _compute_modality_embeddings(self, F, epoch, device):
        X = torch.zeros((self.num_nodes, self.modality_out_dim),
                        dtype=torch.float32)
        offset = 0
        for modality, F_set in F:
            if modality not in self.modality_modules.keys():
                continue

            num_sets = len(F_set)  # number of encoding sets for this datatype
            for i, (encodings, batches) in enumerate(F_set):
                module, seq_length, out_dim = self.modality_modules[modality][i]
                module.to(device)

                num_batches = len(batches)
                for j, (batch_encoding_idx, batch_node_idx) in enumerate(batches):
                    if modality in ["xsd.string", "xsd.anyURI", "ogc.wktLiteral"]:
                        # encodings := list of sparse coo matrices
                        batch = itemgetter(*batch_encoding_idx)(encodings)
                        if type(batch) is not tuple:  # single sample
                            batch = (batch,)

                        time_dim = 1 # if modality == "xsd.string" else 0  ## uncomment for RNN
                        batch = collate_zero_padding(batch,
                                                     time_dim,
                                                     min_padded_length=seq_length)

                        batch = scipy_sparse_list_to_pytorch_sparse(batch)
                        batch = batch.to_dense()
                    else:
                        # encodings := numpy array
                        batch = encodings[batch_encoding_idx]
                        batch = torch.as_tensor(batch)

                    # forward pass
                    batch = batch.to(device)
                    # compute gradients on batch 
                    if epoch > -1 and (epoch-1) % num_batches == j:
                        logger.debug(" {} (set {} / {}) - batch {} / {} +grad".format(modality,
                                                                           i+1, num_sets,
                                                                           j+1, num_batches))
                        out_dev = module(batch)
                    else:
                        logger.debug(" {} (set {} / {}) - batch {} / {} -grad".format(modality,
                                                                           i+1, num_sets,
                                                                           j+1, num_batches))
                        with torch.no_grad():
                            out_dev = module(batch)

                    X[batch_node_idx, offset:offset+out_dim] = out_dev.to('cpu')

                offset += out_dim

        return X

    def init(self):
        # reinitialze all weights
        for module in self.module_dict.values():
            #if type(module) in (ImageCNN, CharCNN, RGCN, RNN):
            if type(module) in (ImageCNN, TCNN, RGCN):
                module.init()
            else:
                raise NotImplementedError
