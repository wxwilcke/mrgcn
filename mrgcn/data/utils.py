#!/usr/bin/python3

import logging
from os import access, F_OK, R_OK, W_OK
from os.path import split
import torch


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

def scipy_sparse_to_pytorch_sparse(sp_input, device=None):
    if device is torch.device("cuda"):
        return torch.cuda.sparse.FloatTensor(torch.LongTensor([sp_input.nonzero()[0],
                                                               sp_input.nonzero()[1]]),
                                             torch.Tensor(sp_input.data),
                                             sp_input.shape)
    else:
        return torch.sparse.FloatTensor(torch.LongTensor([sp_input.nonzero()[0],
                                                          sp_input.nonzero()[1]]),
                                        torch.Tensor(sp_input.data),
                                        sp_input.shape)

