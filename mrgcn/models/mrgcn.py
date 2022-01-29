#!/usr/bin/env python

import logging
from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn

from mrgcn.data.utils import (collate_zero_padding,
                              getNeighboursSparse,
                              scipy_sparse_list_to_pytorch_sparse,
                              scipy_sparse_to_pytorch_sparse)
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
        Mini Batch support

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

    def forward(self, X, A, batch_idx, device=None):
        if batch_idx is ...:
            # full batch
            return self._forward_full_batch(X, A, device)

        return self._forward_mini_batch(X, A, batch_idx, device)

    def _forward_full_batch(self, X, A, device=None):
        X, F = X[0], X[1:]

        # compute and concat modality-specific embeddings
        if self.compute_modality_embeddings:
            batch_idx = np.arange(self.num_nodes)  # full batch

            XF = self._compute_modality_embeddings(F, batch_idx, device)
            X = torch.cat([X,XF], dim=1)

        A = scipy_sparse_to_pytorch_sparse(A)

        # Forward pass through graph convolution layers
        self.rgcn.to(device)
        X_dev = X.to(device)
        A_dev = A.to(device)

        X_dev = self.rgcn(X_dev, A_dev, ...)

        return X_dev

    def _forward_mini_batch(self, X, A, batch_idx, device=None):
        """ Mini batch MR-GCN.

            For each batch, determine which neighbours are needed to compute
            the nodes in the batch, and continue only with the necessary slices 
            of X and A. Do all this here to reduce the memory requirements for
            the GPU. 

            NB: mini batching is currently only implemented for at most two
                R-GCN layers.
        """
        X, F = X[0], X[1:]

        # global node indices of the neighbours of the batch nodes
        # filter batch nodes to prevent computing them twice
        neighbours_idx = getNeighboursSparse(A, batch_idx)
        neighbours_unseen_idx = [i for i in neighbours_idx
                                 if i not in batch_idx]

        # embedding idx for non-input layer
        # filter nodes which we don't need to compute batch
        H_node_idx = np.concatenate([batch_idx, neighbours_unseen_idx])
        H_idx_filtered = [i for i, v in enumerate(H_node_idx)
                          if v in neighbours_idx]
        H_node_idx = H_node_idx[H_idx_filtered]

        # prepare A slices
        A_batch = A[batch_idx]
        A_neighbours_unseen = A[neighbours_unseen_idx]

        A_batch = scipy_sparse_to_pytorch_sparse(A_batch)
        A_neighbours_unseen = scipy_sparse_to_pytorch_sparse(A_neighbours_unseen)

        # subset of X relevant for batch
        X_batch_idx = None
        X_batch = None
        depth2neighbours_idx = list()
        if self.compute_modality_embeddings:
        #if not self.rgcn.layers['layer_0'].featureless:
            # assume layer_0 is input layer
            if len(neighbours_unseen_idx) > 0:
                depth2neighbours_idx = getNeighboursSparse(A, neighbours_unseen_idx)
                
            X_batch_idx = set.union(set(neighbours_idx),
                                    set(depth2neighbours_idx),
                                    set(H_node_idx))
            X_batch_idx = sorted(list(X_batch_idx))

            # compute and concat modality-specific embeddings
            XF = self._compute_modality_embeddings(F, X_batch_idx,
                                                   device)

            #logger.debug(" Merging structure and node features")
            X_batch_idx = torch.LongTensor(X_batch_idx)
            X_batch = X[X_batch_idx]

            X_batch = torch.cat([X_batch, XF], dim=1)

            nodes_needed = len(X_batch_idx)
            logger.debug("Computing embeddings for %d / %d nodes" %
                         (nodes_needed, self.num_nodes))
        else: 
            nodes_needed = len(H_node_idx)
            logger.debug("Computing embeddings for %d / %d nodes" %
                         (nodes_needed, self.num_nodes))
        
        # wrap in list for easy transfer
        neighbours = [torch.LongTensor(neighbours_idx),
                      torch.LongTensor(depth2neighbours_idx),
                      torch.LongTensor(H_idx_filtered),
                      torch.LongTensor(H_node_idx)]

        # move everything to device
        self.rgcn.to(device)
        X_batch_dev = None if X_batch is None else X_batch.to(device)
        A_batch_dev = A_batch.to(device)
        A_neighbours_unseen_dev = A_neighbours_unseen.to(device)
        X_batch_idx_dev = None if X_batch_idx is None else X_batch_idx.to(device)
        neighbours_dev = [t.to(device) for t in neighbours]

        X_dev = self.rgcn(X_batch_dev, 
                          A_batch_dev,
                          X_batch_idx_dev,
                          A_neighbours_unseen_dev,
                          neighbours_dev)

        return X_dev

    def _compute_modality_embeddings(self, F, batch_idx, device):
        batch_num_nodes = len(batch_idx)
        X = torch.zeros((batch_num_nodes, self.modality_out_dim),
                        dtype=torch.float32)
        offset = 0
        for modality, F_set in F:
            if modality not in self.modality_modules.keys():
                continue

            num_sets = len(F_set)  # number of encoding sets for this datatype
            num_features = 0
            for i, encoding_set in enumerate(F_set):
                module, seq_length, out_dim = self.modality_modules[modality][i]
                module.to(device)

                encodings, node_idx, _ = encoding_set
                F_batch_idx = np.where(np.in1d(node_idx, batch_idx))[0]
                if len(F_batch_idx) <= 0:
                    # no nodes in this batch have this modality
                    offset += out_dim
                    continue

                num_features += len(F_batch_idx)
                #batch_idx_map = np.array([j for j in range(len(batch_idx))
                #                         if batch_idx[j] in node_idx])
                batch_idx_map = np.where(np.in1d(batch_idx, node_idx))[0]
                if modality in ["xsd.string", "xsd.anyURI", "ogc.wktLiteral"]:
                    # encodings := list of sparse coo matrices
                    batch = itemgetter(*F_batch_idx)(encodings)
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
                    batch = encodings[F_batch_idx]
                    batch = torch.as_tensor(batch)

                # forward pass
                batch = batch.to(device)
                # compute gradients on batch 
                logger.debug(" {} (set {} / {})".format(modality,
                                                        i+1,
                                                        num_sets))
                out_dev = module(batch)

                X[batch_idx_map, offset:offset+out_dim] = out_dev.to('cpu')

                offset += out_dim

            logger.debug("Added %d / %d feature(s) for datatype %s" % (num_features,
                                                                       batch_num_nodes,
                                                                       modality))

        return X
