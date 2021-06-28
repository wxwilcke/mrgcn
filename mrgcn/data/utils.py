#!/usr/bin/python3

from itertools import cycle
import logging
import os
from os import access, F_OK, R_OK, W_OK
from os.path import split
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.sparse as sp

from mrgcn.encodings.graph_features import (construct_feature_matrix,
                                            features_included)


logger = logging.getLogger(__name__)

def is_readable(filename):
    path = split(filename)[0]
    if not access(path, F_OK):
        raise OSError(":: Path does not exist: {}".format(path))
    elif not access(path, R_OK):
        raise OSError(":: Path not readable by user: {}".format(path))

    return True

def is_writable(filename):
    path = split(filename)[0]
    if not access(path, F_OK):
        raise OSError(":: Path does not exist: {}".format(path))
    elif not access(path, W_OK):
        raise OSError(":: Path not writeable by user: {}".format(path))

    return True

def is_gzip(filename):
    return True if filename.endswith('.gz') else False

def set_seed(seed=-1):
    if seed < 0:
        seed = np.random.randint(0, 2**32-1)

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.debug("Setting seed to {}".format(seed))

def scipy_sparse_list_to_pytorch_sparse(sp_inputs):
    return torch.stack([scipy_sparse_to_pytorch_sparse(sp) for sp in sp_inputs],
                       dim = 0)

def scipy_sparse_to_pytorch_sparse(sp_input):
    return torch.sparse_coo_tensor(torch.LongTensor([sp_input.nonzero()[0],
                                                     sp_input.nonzero()[1]]),
                                   torch.Tensor(sp_input.data),
                                   sp_input.shape,
                                   dtype=torch.float32)

class SparseDataset(Dataset):
    n = 0

    def __init__(self, sp_input):
        # sp_input := a list with sparse coo matrices
        self.n = len(sp_input)
        self.sp = sp_input

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.as_tensor(self.sp[idx].todense())

def collate_zero_padding(batch, time_dim, max_batch_length=999,
                               min_padded_length=3):
    """ batch := a list with sparse coo matrices

        time_dim should be 0 for RNN and 1 for temporal CNN
        min_padded_length >= 3 to allow smallest CNN to support it
    """
    batch_padded = list()

    max_length = 0
    for seq in batch:
        if seq.shape[time_dim] > max_length:
            max_length = seq.shape[time_dim]
    max_length = min(max_length, max_batch_length)
    padded_length = max(min_padded_length, max_length)

    for seq in batch:
        shape = (seq.shape[0], padded_length) if time_dim == 1\
                else (padded_length, seq.shape[1])

        a = sp.coo_matrix((seq.data, (seq.row, seq.col)),
                          shape=shape, dtype=np.float32)
        batch_padded.append(a)

    return batch_padded

def collate_repetition_padding(batch, time_dim, max_batch_length=999,
                               min_padded_length=3):
    """ batch := a list with sparse coo matrices

        time_dim should be 0 for RNN and 1 for temporal CNN
    """
    batch_padded = list()

    max_length = 0
    for seq in batch:
        if seq.shape[time_dim] > max_length:
            max_length = seq.shape[time_dim]
    max_length = min(max_length, max_batch_length)
    padded_length = max(min_padded_length, max_length)

    for seq in batch:
        feature_idc = seq.row if time_dim == 1 else seq.col
        sequence_idc = seq.col if time_dim == 1 else seq.row

        data = list(seq.data)
        feat_idc = list(feature_idc)
        seq_idc = list(sequence_idc)

        seq_length = seq.shape[time_dim]
        unfilled = padded_length - seq_length
        if unfilled > 0:
            c_data = cycle(seq.data)
            c_feat = cycle(feature_idc)

            i = 0
            t = 0
            while unfilled > 0:
                j = 0
                for c in sequence_idc[i:]:
                    if c != sequence_idc[i]:
                        break
                    j += 1

                data.extend([next(c_data) for _ in range(j)])
                feat_idc.extend([next(c_feat) for _ in range(j)])
                seq_idc.extend([seq_length+t for _ in range(j)])

                i += j
                if i >= len(sequence_idc):
                    i = 0
                t += 1
                unfilled -= 1
        elif unfilled < 0:  # sequence exceeds max length
            sequence_idc_rev = [v for v in sequence_idc]
            sequence_idc_rev.reverse()
            i = 0
            k = 0
            while unfilled < 0:
                j = 0
                for c in sequence_idc_rev[i:]:
                    if c != sequence_idc_rev[i]:
                        break
                    j += 1

                i += j
                k += j
                if i >= len(sequence_idc):
                    i = 0
                unfilled += 1

            data = data[:-k]
            feat_idc = feat_idc[:-k]
            seq_idc = seq_idc[:-k]
        else:
            pass  # already at desired size

        coordinates = (feat_idc, seq_idc) if time_dim == 1\
                else (seq_idc, feat_idc)
        shape = (seq.shape[1-time_dim], padded_length) if time_dim == 1\
                else (padded_length, seq.shape[1-time_dim])

        a = sp.coo_matrix((data, coordinates),
                          shape=shape, dtype=np.float32)
        batch_padded.append(a)

    return batch_padded

def setup_features(F, num_nodes, featureless, config):
    X_width = 0  # number of columns in X
    X = [torch.empty((num_nodes, X_width), dtype=torch.float32)]  # dummy

    modules_config = list()
    optimizer_config = list()
    if not featureless:
        features_enabled = features_included(config)
        logging.debug("Features included: {}".format(", ".join(features_enabled)))
        for datatype in features_enabled:
            if datatype in F.keys():
                logger.debug("Found {} encoding set(s) for datatype {}".format(
                    len(F[datatype]),
                    datatype))

        # create batched  representations for neural encodings
        feature_configs = config['graph']['features']
        features, modules_config, optimizer_config, feat_width = construct_feature_matrix(F,
                                                                                          features_enabled,
                                                                                          feature_configs)
        X_width += feat_width
        X.extend(features)

    return (X, X_width, modules_config, optimizer_config)
