#!/usr/bin/env python

import logging
from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn

from mrgcn.data.utils import (collate_zero_padding,
                              scipy_sparse_list_to_pytorch_sparse,
                              scipy_sparse_to_pytorch_sparse)
from mrgcn.models.fully_connected import FC
from mrgcn.models.imagecnn import ImageCNN
#from mrgcn.models.rnn import RNN
from mrgcn.models.rgcn import RGCN
from mrgcn.models.temporal_cnn import TCNN
from mrgcn.data.batch import MiniBatch


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

        self.num_nodes = num_nodes  # != labelled nodes
        self.p_dropout = p_dropout
        self.module_dict = nn.ModuleDict()

        # add embedding layers
        self.modality_modules = dict()
        self.modality_out_dim = 0
        self.compute_modality_embeddings = False
        h, i, j, k = 0, 0, 0, 0
        for datatype, args in embedding_modules:
            module = None
            dim_out = -1
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

    def forward(self, batch, device=None):
        if isinstance(batch, MiniBatch):
            return self._forward_mini_batch(batch, device)

        # full batch
        return self._forward_full_batch(batch, device)

    def _forward_full_batch(self, batch, device=None):
        X, F = batch.X[0], batch.X[1:]

        # compute and concat modality-specific embeddings
        if self.compute_modality_embeddings:
            # expects a natural ordering of nodes; same as X
            batch_idx = np.arange(self.num_nodes)  # full batch

            XF = self._compute_modality_embeddings(F, batch_idx, device)
            X = torch.cat([X,XF], dim=1)

        A = batch.A

        # Forward pass through graph convolution layers
        self.rgcn.to(device)
        X_dev = X.to(device)
        A_dev = A.to(device)

        X_dev = self.rgcn(X_dev, A_dev)

        return X_dev

    def _forward_mini_batch(self, batch, device):
        """ Mini batch MR-GCN.
        """

        X_dev = None
        if self.compute_modality_embeddings:
            # compute the initial embeddings to use as input to the R-GCN.
            # this is only necessary for the most-distant nodes, since 
            # nodes closer to the batch nodes will use the embeddings computed
            # by the previous graph convvolution layers.
            if batch.X is not None:
                X, F = batch.X[0], batch.X[1:]

                # most distant nodes
                batch_idx = batch.A.neighbours[-1]

                XF = self._compute_modality_embeddings(F, batch_idx, device)
                X = torch.cat([X,XF], dim=1)

                X_dev = X.to(device)

        # Forward pass through graph convolution layers
        self.rgcn.to(device)
        # TODO: move batch to device?
        #A_dev = A.to(device)
        A_dev = batch.A
        X_dev = self.rgcn(X_dev, A_dev)

        return X_dev

    def _compute_modality_embeddings(self, F, batch_idx, device):
        batch_num_nodes = len(batch_idx)
        X = torch.zeros((batch_num_nodes, self.modality_out_dim),
                        dtype=torch.float32)
        offset = 0
        for modality, F_set in F:  # F_set is actually alist
            if modality not in self.modality_modules.keys():
                continue

            num_sets = len(F_set)  # number of encoding sets for this datatype
            num_features = 0  # number of added node features for this modality
            for i, encoding_set in enumerate(F_set):
                module, _, out_dim = self.modality_modules[modality][i]
                module.to(device)

                encodings, node_idx, _ = encoding_set
                common_nodes = np.intersect1d(node_idx, batch_idx)
                if len(common_nodes) <= 0:
                    # no nodes in this batch have this modality
                    offset += out_dim
                    continue

                num_features += len(common_nodes)

                # find indices for common nodes
                F_batch_mask = np.in1d(node_idx, common_nodes)
                X_batch_mask = np.in1d(batch_idx, common_nodes)

                # forward pass
                data = encodings[F_batch_mask].float()
                data = data.to(device)
                # compute gradients on batch 
                logger.debug(" {} (set {} / {})".format(modality,
                                                        i+1,
                                                        num_sets))
                out_dev = module(data)

                X[X_batch_mask, offset:offset+out_dim] = out_dev.to('cpu')

                offset += out_dim

            logger.debug("Added %d / %d feature(s) for datatype %s" % (num_features,
                                                                       batch_num_nodes,
                                                                       modality))

        return X

#    def _compute_modality_embeddings(self, F, batch_idx, device):
#        batch_num_nodes = len(batch_idx)
#        X = torch.zeros((batch_num_nodes, self.modality_out_dim),
#                        dtype=torch.float32)
#        offset = 0
#        for modality, F_set in F:  # F_set is actually alist
#            if modality not in self.modality_modules.keys():
#                continue
#
#            num_sets = len(F_set)  # number of encoding sets for this datatype
#            num_features = 0  # number of added node features for this modality
#            for i, encoding_set in enumerate(F_set):
#                module, seq_length, out_dim = self.modality_modules[modality][i]
#                module.to(device)
#
#                encodings, node_idx, _ = encoding_set
#                common_nodes = np.intersect1d(node_idx, batch_idx)
#                if len(common_nodes) <= 0:
#                    # no nodes in this batch have this modality
#                    offset += out_dim
#                    continue
#
#                # find indices for common nodes
#                F_batch_idx = np.where(np.in1d(node_idx, common_nodes))[0]
#                X_batch_idx = np.where(np.in1d(batch_idx, common_nodes))[0]
#
#                num_features += len(common_nodes)
#                if modality in ["xsd.string", "xsd.anyURI", "ogc.wktLiteral"]:
#                    # encodings := list of sparse coo matrices
#                    batch = itemgetter(*F_batch_idx)(encodings)
#                    if type(batch) is not tuple:  # single sample
#                        batch = (batch,)
#
#                    time_dim = 1 # if modality == "xsd.string" else 0  ## uncomment for RNN
#                    batch = collate_zero_padding(batch,
#                                                 time_dim,
#                                                 min_padded_length=seq_length)
#
#                    batch = scipy_sparse_list_to_pytorch_sparse(batch)
#                    batch = batch.to_dense()
#                else:
#                    # encodings := numpy array
#                    batch = encodings[F_batch_idx]
#                    batch = torch.as_tensor(batch)
#
#                # forward pass
#                batch = batch.to(device)
#                # compute gradients on batch 
#                logger.debug(" {} (set {} / {})".format(modality,
#                                                        i+1,
#                                                        num_sets))
#                out_dev = module(batch)
#
#                X[X_batch_idx, offset:offset+out_dim] = out_dev.to('cpu')
#
#                offset += out_dim
#
#            logger.debug("Added %d / %d feature(s) for datatype %s" % (num_features,
#                                                                       batch_num_nodes,
#                                                                       modality))
#
#        return X

#def mini_batch_package(X, A, batch_idx, num_layers):
#    """ Mini-batch package
#
#        Gather everything necessary to compute batch embeddings.
#        Assume A to be a sparse scipy matrix
#    """
#    payload = list()    
#
#    sample_idx = batch_idx  # nodes to compute the embeddings for at layer i
#    for i in range(1, num_layers+1):       
#        pack = dict()
#                
#        # slices of A belonging to the sample nodes at layer i
#        A_samples = A[sample_idx]
#        A_samples = scipy_sparse_to_pytorch_sparse(A_samples)
#        
#        # neighbours of the sample nodes at layer i
#        neighbours_idx = getNeighboursSparse(A, sample_idx)
#                
#        pack["A_samples"] = A_samples
#        pack["neighbours_idx"] = neighbours_idx
#        if i >= num_layers and X is not None:
#            # features of neighbours
#            # only necessary for most-distant nodes
#            pack["X_neighbours"] = subset_data(X, neighbours_idx)
#        else:
#            pack["X_neighbours"] = None
#            
#        sample_idx = neighbours_idx
#            
#        payload.append(pack)
#            
#    return payload



## TODO: move X out of batch; move out epoch loop
#class Batch:
#    idx = list()  # nodes to compute embeddings for
#    neighbours = list()  # nodes necessary to compute embeddings
#    A = list()  # adjacency matrix slices necessary to compute embeddings
#    X = None  # raw values of outer neighbours if node features are included
#
#    def __init__(self, X, A, batch_idx, num_layers):
#        self.idx = batch_idx
#
#        self._populate(X, A, num_layers)
#
#    def _populate(self, X, A, num_layers):
#        sample_idx = self.idx  # nodes to compute the embeddings for at layer i
#        for i in range(1, num_layers+1):       
#            # slices of A belonging to the sample nodes at layer i
#            A_samples = A[sample_idx]
#            A_samples = scipy_sparse_to_pytorch_sparse(A_samples)
#            
#            # neighbours of the sample nodes at layer i
#            neighbours_idx = getNeighboursSparse(A, sample_idx)
#                    
#            self.A.append(A_samples)
#            self.neighbours.append(neighbours_idx)
#            if i >= num_layers and X is not None:
#                # features of neighbours
#                # only necessary for most-distant nodes
#                self.X = subset_data(X, neighbours_idx)
#                
#            sample_idx = neighbours_idx
#
#    def to(self, device):
#        # TODO
#        pass
