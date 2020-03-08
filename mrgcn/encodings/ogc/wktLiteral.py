#!/usr/bin/python3

import logging
from re import fullmatch

from deep_geometry import vectorizer as gv
import numpy as np
from rdflib.term import Literal, URIRef
from rdflib import Namespace
import scipy.sparse as sp


_REGEX_COORDINATE = "\d\.?\d* \d\.?\d*"
_REGEX_WKTPOINT = "POINT\s?\(" + _REGEX_COORDINATE + "\)"
_REGEX_WKTPOLYGON = "POLYGON\s?\((\(" + _REGEX_COORDINATE + "[,\s" + _REGEX_COORDINATE + "]*\)(,\s)?)*\)"
_REGEX_WKTLITERAL = _REGEX_WKTPOINT + "|" + _REGEX_WKTPOLYGON

_OGC_NAMESPACE = Namespace(URIRef("http://www.opengis.net/ont/geosparql#"))

_MAX_POINTS = 64
_GEOVECTORIZER_VEC_LENGTH = 7

logger = logging.getLogger(__name__)

def generate_features(nodes_map, node_predicate_map, config, time_dim=1):
    """ Generate features for OGC WKT literals

    time_dim == 0 for RNN, 1 for CNN

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param config: configuration dictionary
    :returns: numpy array N x C;
                    N :- number of nodes
                    C :- number of columns for this feature embedding
    """
    logger.debug("Generating wktLiteral features")
    C = 16  # number of items per feature

    n = len(nodes_map)
    node_idx = np.zeros(shape=(n), dtype=np.int32)
    vec_length_map = list()
    data = list()

    m = 0
    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype.neq(_OGC_NAMESPACE.wktLiteral):
            continue

        value = str(node)  ## empty value bug workaround
        try:
            vec = gv.vectorize_wkt(value)[:_MAX_POINTS,:]
        except:
            continue

        vec_length = vec.shape[0]
        if vec_length <= 0:
            continue

        # create matrix with time dimension over rows 
        sp_rows, sp_cols = np.where(vec > 0.0)
        if time_dim == 0:
            a = sp.csr_matrix((vec[(sp_rows, sp_cols)], (sp_rows, sp_cols)),
                              shape=(vec_length, _GEOVECTORIZER_VEC_LENGTH),
                              dtype=np.float64)
        else:  # time_dim == 1
            a = sp.csr_matrix((vec[(sp_rows, sp_cols)], (sp_cols, sp_rows)),
                              shape=(_GEOVECTORIZER_VEC_LENGTH, vec_length),
                              dtype=np.float64)

        data.append(a)
        vec_length_map.append(vec_length)
        node_idx[m] = i
        m += 1

    logger.debug("Generated {} unique wktLiteral features".format(m))

    if m <= 0:
        return None

    # normalization
    sc = GeomScalerSparse(time_dim)
    means = sc.fit(data)
    data = sc.transform(data, means)

    return [[data, node_idx[:m], C, vec_length_map, 1]]

def validate(value):
    return fullmatch(_REGEX_WKTLITERAL, value)


### add sparse matrix support for geoscaler ###

class GeomScalerSparse:
    FULL_STOP_INDEX = -1
    def __init__(self, time_dim=0):
        self.scale_factor = 1.
        self.time_dim = time_dim

    def fit(self, geometry_vectors):
        means = [self.localized_mean(v) for v in geometry_vectors]
        min_maxs = list()

        for index, geometry in enumerate(geometry_vectors):
            full_stop_point_index = self.get_full_stop_index(geometry)

            x_and_y_coords = geometry[:full_stop_point_index, :2]\
                    if self.time_dim == 0 else geometry[:2, :full_stop_point_index]
            min_maxs.append([
                np.min(x_and_y_coords - means[index]),
                np.max(x_and_y_coords - means[index])
            ])

        self.scale_factor = np.std(min_maxs)

        return means

    def transform(self, geometry_vectors, means):
        localized = list()
        for index, geometry in enumerate(geometry_vectors):
            stop_index = self.get_full_stop_index(geometry) + 1
            geometry_copy = geometry.copy().tolil()
            if self.time_dim == 0:
                geometry_copy[:stop_index, :2] -= means[index]
                geometry_copy[:stop_index, :2] /= self.scale_factor
            else:
                geometry_copy[:2, :stop_index] -= means[index]
                geometry_copy[:2, :stop_index] /= self.scale_factor

            localized.append(geometry_copy.tocoo())

        return localized

    def get_full_stop_index(self, geometry_vector):
        full_stop_slice = geometry_vector[:, self.FULL_STOP_INDEX]\
                if self.time_dim == 0 else geometry_vector[self.FULL_STOP_INDEX, :]
        full_stop_point_index = sp.find(full_stop_slice == 1.0)[self.time_dim][0]

        if full_stop_point_index == 0:
            full_stop_point_index = -1

        return full_stop_point_index

    def localized_mean(self, geometry_vector):
        full_stop_point_index = self.get_full_stop_index(geometry_vector)
        geom_mean = geometry_vector[:full_stop_point_index, :2].mean(axis=0)\
                if self.time_dim == 0 else geometry_vector[:2, :full_stop_point_index].mean(axis=1)
        return geom_mean

