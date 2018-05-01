#!/usr/bin/python3

from io import BytesIO
import pickle
import logging
import os
import tarfile

import numpy as np
import keras.engine.training as ket
import scipy.sparse as sp


logger = logging.getLogger(__name__)

def store(path, files, names):
    assert len(files) == len(names)
    logger.info("Storing to file: {}".format(path))
    with tarfile.TarFile(path, 'w') as tf:
        for i, f in enumerate(files):
            if type(f) is sp.csr.csr_matrix:
                _store_csr(tf, f, names[i])
            elif type(f) is np.ndarray:
                _store_nda(tf, f, names[i])
            elif type(f) is ket.Model:
                _store_kfm(tf, f, names[i])
            elif type(f) is dict:
                _store_dict(tf, f, os.path.join('dict', names[i]))
            elif type(f) is list:
                _store_list(tf, f, os.path.join('list', names[i]))
            else:
                _store_py(tf, f, names[i])

def _add_to_tar(tf, f, name, size):
    info = tarfile.TarInfo(name=name)
    info.size=size
    tf.addfile(tarinfo=info, fileobj=f)

def _store_list(tf, f, name):
    for i, item in enumerate(f):
        i = str(i)
        if type(item) is list:
            _store_list(tf, item, os.path.join(name, 'list'))
        elif type(item) is sp.csr.csr_matrix:
            _store_csr(tf, item, os.path.join(name, i))
        elif type(item) is np.ndarray:
            _store_nda(tf, item, os.path.join(name, i))
        elif type(item) is ket.Model:
            _store_kfm(tf, item, os.path.join(name, i))
        else:
            _store_py(tf, item, os.path.join(name, i))

def _store_csr(tf, f, name):
    buff = BytesIO()
    np.savez(buff, data=f.data, indices=f.indices, indptr=f.indptr, shape=f.shape)

    size = len(buff.getbuffer())
    buff.seek(0)

    _add_to_tar(tf, buff, name+'.npz', size)

def _store_nda(tf, f, name):
    buff = BytesIO()
    np.save(buff, f)

    size = len(buff.getbuffer())
    buff.seek(0)

    _add_to_tar(tf, buff, name+'.npy', size)

def _store_kfm(tf, f, name):
    # h5py does not support file objects
    path = './.{}.tmp'.format(name)
    f.save(path)
    tf.add(path, arcname='{}.h5'.format(name))
    os.remove(path)

def _store_dict(tf, f, name):
    if type(f) is not dict:
        if type(f) is sp.csr.csr_matrix:
            _store_csr(tf, f, name)
        elif type(f) is np.ndarray:
            _store_nda(tf, f, name)
        elif type(f) is ket.Model:
            _store_kfm(tf, f, name)
        else:
            _store_py(tf, f, name)

        return

    for k,v in f.items():
        _store_dict(tf, v, os.path.join(name,k))

def _store_py(tf, f, name):
    buff = BytesIO()
    pickle.dump(f, buff, protocol=-1)
    
    size = len(buff.getbuffer())
    buff.seek(0)

    _add_to_tar(tf, buff, name+'.pkl', size)

def _add_dir(tf, name):
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    tf.addfile(info)
