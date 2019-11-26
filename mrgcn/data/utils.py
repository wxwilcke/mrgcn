#!/usr/bin/python3

import logging
from os import access, F_OK, R_OK, W_OK
from os.path import split

import numpy as np
import torch
from torch.utils.data import Dataset


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

def scipy_sparse_to_pytorch_sparse(sp_input):
    t = torch.empty(0)
    if type(sp_input) is list:
        n = len(sp_input)
        nrows, ncols = sp_input[0].shape
        row_idx = [i for a in sp_input for i in a.nonzero()[0]]
        t = torch.sparse_coo_tensor([np.repeat(range(n), ncols),
                                     row_idx,
                                     np.tile(range(ncols), n)],
                                    np.repeat([1.0], repeats=n*ncols),
                                    size=(n, nrows, ncols),
                                    dtype=torch.float32)
    else:
        t = torch.sparse_coo_tensor(torch.LongTensor([sp_input.nonzero()[0],
                                                      sp_input.nonzero()[1]]),
                                    torch.Tensor(sp_input.data),
                                    sp_input.shape,
                                    dtype=torch.float32)

    return t

class SparseDataset(Dataset):
    n = 0

    def __init__(self, sp_input):
        self.n = sp_input.size(0)
        self.sp = sp_input

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.sp[idx].to_dense()
