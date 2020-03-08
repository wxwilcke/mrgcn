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
                                            features_included,
                                            merge_sparse_encodings_sets)
from mrgcn.tasks.utils import (mkbatches,
                               mkbatches_varlength,
                               remove_outliers,
                               trim_outliers)


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
    C = 0  # number of columns in X
    X = [torch.empty((num_nodes,C), dtype=torch.float32)]
    modules_config = list()
    if not featureless:
        features_enabled = features_included(config)
        logging.debug("Features included: {}".format(", ".join(features_enabled)))

        X, F = construct_feature_matrix(F, features_enabled, num_nodes,
                                        config['graph']['features'])
        C += X.shape[1]
        X = [torch.as_tensor(X)]

        # determine configurations
        for datatype in features_enabled:
            if datatype not in F.keys():
                continue

            logger.debug("Found {} encoding set(s) for datatype {}".format(
                len(F[datatype]),
                datatype))

            if datatype not in ['xsd.string', 'ogc.wktLiteral', 'blob.image']:
                continue

            feature_configs = config['graph']['features']
            feature_config = next((conf for conf in feature_configs
                                   if conf['datatype'] == datatype),
                                  None)

            # preprocess
            encoding_sets = F.pop(datatype, list())
            weight_sharing = feature_config['share_weights']
            if weight_sharing and datatype == "xsd.string":
                # note: images and geometries always share weights atm
                logger.debug("weight sharing enabled for {}".format(datatype))
                encoding_sets = merge_sparse_encodings_sets(encoding_sets)

            for encodings, node_idx, c, seq_lengths, nsets in encoding_sets:
                if datatype in ["xsd.string"]:
                    # stored as list of arrays
                    feature_dim = 0
                    feature_size = encodings[0].shape[feature_dim]

                    model_size = "M"  # medium, seq length >= 12
                    if not weight_sharing or nsets <= 1:
                        seq_length_min = min(seq_lengths)
                        if seq_length_min < 20:
                            model_size = "S"
                        elif seq_length_min < 50:
                            model_size = "M"
                        else:
                            model_size = "L"

                    modules_config.append((datatype, (feature_config['passes_per_batch'],
                                                      feature_size,
                                                      c,
                                                      model_size)))
                if datatype in ["ogc.wktLiteral"]:
                    # stored as list of arrays
                    feature_dim = 0  # set to 1 for RNN
                    feature_size = encodings[0].shape[feature_dim]
                    modules_config.append((datatype, (feature_config['passes_per_batch'],
                                                      feature_size,
                                                      c)))
                if datatype in ["blob.image"]:
                    # stored as tensor
                    modules_config.append((datatype, (feature_config['passes_per_batch'],
                                                      encodings.shape[1:],
                                                      c)))

                C += c

            # deal with outliers?
            if datatype in ["ogc.wktLiteral", "xsd.string"]:
                if feature_config['remove_outliers']:
                    encoding_sets = [remove_outliers(*f) for f in encoding_sets]
                if feature_config['trim_outliers']:
                    feature_dim = 0  # set to 1 for RNN
                    encoding_sets = [trim_outliers(*f, feature_dim) for f in encoding_sets]

            nepoch = config['model']['epoch']
            encoding_sets_batched = list()
            for f in encoding_sets:
                if datatype == "blob.image":
                    encoding_sets_batched.append((f, mkbatches(*f,
                                                      nepoch=nepoch,
                                                      passes_per_batch=feature_config['passes_per_batch'])))
                elif datatype == "ogc.wktLiteral":
                    encoding_sets_batched.append((f, mkbatches_varlength(*f,
                                                                nepoch=nepoch,
                                                                passes_per_batch=feature_config['passes_per_batch'])))
                elif datatype == "xsd.string":
                    encoding_sets_batched.append((f, mkbatches_varlength(*f,
                                                                nepoch=nepoch,
                                                                passes_per_batch=feature_config['passes_per_batch'])))

            X.append((datatype, encoding_sets_batched))

    return (X, C, modules_config)
