#!/usr/bin/python3

from io import BytesIO
import pickle
import logging
import os
import tarfile

import numpy as np
import keras.models as km
import scipy.sparse as sp


logger = logging.getLogger(__name__)

def read(path):
    content = []
    names = []
    logger.info("Loading from file: {}".format(path))
    with tarfile.TarFile(path, 'r') as tf:
        members = set(tf.getnames())

        # check for structures
        paths = {member for member in members if '/' in member}
        if len(paths) > 0:
            dicts = {path for path in paths if path.split('/')[0] == 'dict' }
            lists = {path for path in paths if path.split('/')[0] == 'list' }

            top_levels = {path.split('/')[1] for path in dicts}
            for top_level in top_levels:
                d_full = [path for path in dicts if path.split('/')[1] == top_level]
                d_base = [path[len(top_level)+1:] for path in d_full]
                content.append(_load_dict(tf, d_full, d_base))
                names.append(top_level)
        
            top_levels = {path.split('/')[1] for path in lists}
            for top_level in top_levels:
                l_full = [path for path in lists if path.split('/')[1] == top_level]
                l_base = [path[len(top_level)+1:] for path in l_full]
                content.append(_load_list(tf, l_full, l_base))
                names.append(top_level)

        # objects
        members -= paths
        for name in members:
            base, ext = os.path.splitext(name)
            if ext == '.npz':
                content.append(_load_csr(tf, name))
                names.append(base)
            elif ext == '.npy':
                content.append(_load_nda(tf, name))
                names.append(base)
            elif ext == '.h5':
                content.append(_load_kfm(tf, name))
                names.append(base)
            else:
                content.append(_load_py(tf, name))
                names.append(base)

    return (content, names)

def _load_csr(tf, name):
    loader = _load_nda(tf, name)
    return sp.csr_matrix((loader['data'], 
                          loader['indices'], 
                          loader['indptr']),
                        shape=loader['shape'], 
                        dtype=np.float32)

def _load_nda(tf, name):
    # https://github.com/numpy/numpy/issues/7989
    buff = BytesIO()
    buff.write(tf.extractfile(name).read())
    buff.seek(0)
    return np.load(buff)

def _load_kfm(tf, name):
    # h5py does not support file objects
    tf.extract(name)
    path = './{}'.format(name)
    model = km.load_model(path)
    os.remove(path)

    return model

def _load_py(tf, name):
    return pickle.load(tf.extractfile(name))

def _load_list(tf, names, paths):
    l = []
    for name, path in zip(names, paths):
        l.append(_load_sublist(tf, name, path.split('/')[1:]))
    
    return l

def _load_sublist(tf, name, path_splitted):
    if len(path_splitted) > 1:
        return _load_sublist(tf, name, path_splitted[1:])
    else:
        filename = path_splitted[0]
        if _ext_of(filename) == '.npz':
            return _load_csr(tf, name)
        elif _ext_of(filename) == '.npy':
            return _load_nda(tf, name)
        elif _ext_of(filename) == '.h5':
            return _load_kfm(tf, name)
        else:
            return _load_py(tf, name)

def _load_dict(tf, names, paths):
    d = {}
    for name, path in zip(names, paths):
        _load_subdict(tf, name, path.split('/')[1:], d)

    return d

def _load_subdict(tf, name, path_splitted, d):
    if len(path_splitted) > 1:
        k = path_splitted[0]
        if k not in d.keys():
            d[k] = {}
        _load_subdict(tf, name, path_splitted[1:], d[k])
        
        return

    filename = path_splitted[0]
    k, ext = os.path.splitext(filename)
    value = None
    if ext == '.npz':
        value = _load_csr(tf, name)
    elif ext == '.npy':
        value = _load_nda(tf, name)
    elif ext == '.h5':
        value = _load_kfm(tf, name)
    else:
        value = _load_py(tf, name)

    d[k] = value

def _ext_of(path):
    return os.path.splitext(path)[-1]
