#!/usr/bin/python3

from io import BytesIO
import pickle
import logging
import os
import tarfile

import numpy as np
import keras.engine.training as ket
import keras.models as km
import scipy.sparse as sp


class Tarball:
    """ Tarball Class
    Create tar archive of numpy, scipy, and keras objects
    """

    tar = None
    _content = {}

    def __init__(self, path=None, mode='r'):
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initiating Tarball")

        if path is None:
            raise ValueError("::No path supplied")

        self.tar = tarfile.open(path, mode)

        if 'r' in mode:
            self.logger.debug("Loading from file: {}".format(path))
            self._content = self.read(path)
        else:
            self.logger.debug("Storing to file: {}".format(path))

    def list_members(self):
        names = set()
        members = set(self.tar.getnames())

        # check for structures
        paths = {member for member in members if '/' in member}
        for path in paths:
            names.add(path.split('/')[1])

        # objects
        members -= paths
        for name in members:
            base, ext = os.path.splitext(name)
            names.add(base)

        return list(names)

    def read(self, path):
        content = []
        names = []
        members = set(self.tar.getnames())

        # check for structures
        paths = {member for member in members if '/' in member}
        if len(paths) > 0:
            dicts = {path for path in paths if path.split('/')[0] == 'dict' }
            lists = {path for path in paths if path.split('/')[0] == 'list' }

            top_levels = {path.split('/')[1] for path in dicts}
            for top_level in top_levels:
                d_full = [path for path in dicts if path.split('/')[1] == top_level]
                d_base = [path[len(top_level)+1:] for path in d_full]
                content.append(self._read_dict(d_full, d_base))
                names.append(top_level)
        
            top_levels = {path.split('/')[1] for path in lists}
            for top_level in top_levels:
                l_full = [path for path in lists if path.split('/')[1] == top_level]
                l_full.sort()  # maintain list order
                l_base = [path[len(top_level)+1:] for path in l_full]
                content.append(self._read_list(l_full, l_base))
                names.append(top_level)

        # objects
        members -= paths
        for name in members:
            base, ext = os.path.splitext(name)
            if ext == '.npz':
                content.append(self._read_csr(name))
                names.append(base)
            elif ext == '.npy':
                content.append(self._read_nda(name))
                names.append(base)
            elif ext == '.h5':
                content.append(self._read_kfm(name))
                names.append(base)
            else:
                content.append(self._read_py( name))
                names.append(base)

        self.logger.debug("Found data structures: {}".format(names))
        return {k:v for k,v in zip(names, content)}

    def store(self, files, names):
        assert len(files) == len(names)
        for i, f in enumerate(files):
            if type(f) is sp.csr.csr_matrix:
                self._store_csr(f, names[i])
            elif type(f) is np.ndarray:
                self._store_nda(f, names[i])
            elif type(f) is ket.Model:
                self._store_kfm(f, names[i])
            elif type(f) is dict:
                self._store_dict(f, os.path.join('dict', names[i]))
            elif type(f) is list:
                self._store_list(f, os.path.join('list', names[i]))
            else:
                self._store_py(f, names[i])

    def get(self, key):
        return self._content[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.tar.close()

    def __len__(self):
        return len(self._content)


### Type-specific readers

    def _read_csr(self, name):
        loader = self._read_nda(name)
        return sp.csr_matrix((loader['data'], 
                              loader['indices'], 
                              loader['indptr']),
                            shape=loader['shape'], 
                            dtype=np.float32)

    def _read_nda(self, name):
        # https://github.com/numpy/numpy/issues/7989
        buff = BytesIO()
        buff.write(self.tar.extractfile(name).read())
        buff.seek(0)
        return np.load(buff)

    def _read_kfm(self, name):
        # h5py does not support file objects
        self.tar.extract(name)
        path = './{}'.format(name)
        model = km.load_model(path)
        os.remove(path)

        return model

    def _read_py(self, name):
        return pickle.load(self.tar.extractfile(name))

    def _read_list(self, names, paths):
        l = []
        for name, path in zip(names, paths):
            l.append(self._read_sublist(name, path.split('/')[1:]))
        
        return l

    def _read_sublist(self, name, path_splitted):
        if len(path_splitted) > 1:
            return self._read_sublist(name, path_splitted[1:])
        else:
            filename = path_splitted[0]
            if self._ext_of(filename) == '.npz':
                return self._read_csr(name)
            elif self._ext_of(filename) == '.npy':
                return self._read_nda(name)
            elif self._ext_of(filename) == '.h5':
                return self._read_kfm(name)
            else:
                return self._read_py(name)

    def _read_dict(self, names, paths):
        d = {}
        for name, path in zip(names, paths):
            self._read_subdict(name, path.split('/')[1:], d)

        return d

    def _read_subdict(self, name, path_splitted, d):
        if len(path_splitted) > 1:
            k = path_splitted[0]
            if k not in d.keys():
                d[k] = {}
            self._read_subdict(name, path_splitted[1:], d[k])
            
            return

        filename = path_splitted[0]
        k, ext = os.path.splitext(filename)
        value = None
        if ext == '.npz':
            value = self._read_csr(name)
        elif ext == '.npy':
            value = self._read_nda(name)
        elif ext == '.h5':
            value = self._read_kfm(name)
        else:
            value = self._read_py(name)

        d[k] = value


### Type-specific writers

    def _store_list(self, f, name):
        for i, item in enumerate(f):
            i = str(i)
            if type(item) is list:
                self._store_list(item, os.path.join(name, 'list'))
            elif type(item) is sp.csr.csr_matrix:
                self._store_csr(item, os.path.join(name, i))
            elif type(item) is np.ndarray:
                self._store_nda(item, os.path.join(name, i))
            elif type(item) is ket.Model:
                self._store_kfm(item, os.path.join(name, i))
            else:
                self._store_py(item, os.path.join(name, i))

    def _store_csr(self, f, name):
        buff = BytesIO()
        np.savez(buff, data=f.data, indices=f.indices, indptr=f.indptr, shape=f.shape)

        size = len(buff.getbuffer())
        buff.seek(0)

        self._add_to_tar(buff, name+'.npz', size)

    def _store_nda(self, f, name):
        buff = BytesIO()
        np.save(buff, f)

        size = len(buff.getbuffer())
        buff.seek(0)

        self._add_to_tar(buff, name+'.npy', size)

    def _store_kfm(self, f, name):
        # h5py does not support file objects
        path = './.{}.tmp'.format(name)
        f.save(path)
        self.tar.add(path, arcname='{}.h5'.format(name))
        os.remove(path)

    def _store_dict(self, f, name):
        if type(f) is not dict:
            if type(f) is sp.csr.csr_matrix:
                self._store_csr(f, name)
            elif type(f) is np.ndarray:
                self._store_nda(f, name)
            elif type(f) is ket.Model:
                self._store_kfm(f, name)
            else:
                self._store_py(f, name)

            return

        for k,v in f.items():
            self._store_dict(v, os.path.join(name,k))

    def _store_py(self, f, name):
        buff = BytesIO()
        pickle.dump(f, buff, protocol=-1)
        
        size = len(buff.getbuffer())
        buff.seek(0)

        self._add_to_tar(buff, name+'.pkl', size)

    def _add_to_tar(self, f, name, size):
        info = tarfile.TarInfo(name=name)
        info.size=size
        self.tar.addfile(tarinfo=info, fileobj=f)

    def _add_dir(self, name):
        info = tarfile.TarInfo(name)
        info.type = tarfile.DIRTYPE
        self.tar.addfile(info)


### Utils

    def _ext_of(self, path):
        return os.path.splitext(path)[-1]
