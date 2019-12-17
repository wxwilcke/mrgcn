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

def generate_features(nodes_map, node_predicate_map, config):
    """ Generate features for OGC WKT literals

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param config: configuration dictionary
    :returns: numpy array N x C;
                    N :- number of nodes
                    C :- number of columns for this feature embedding
    """
    logger.debug("Generating wktLiteral features")
    C = 32  # number of items per feature

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
        if validate(value) is None:
            continue

        vec = gv.vectorize_wkt(value)[:_MAX_POINTS,:]
        vec_length = vec.shape[0]
        if vec_length <= 0:
            continue

        # pad with repetition  - move to preload phase 
        #c = cycle(vec)
        #unfilled = _MAX_POINTS - vec_length
        #if unfilled > 0:
        #    vec.extend([next(c) for _ in range(unfilled)])

        #vec = np.array(vec)

        # create matrix with time dimension over rows (RNN-like)
        sp_rows, sp_cols = np.where(vec > 0.0)
        a = sp.coo_matrix((vec[(sp_rows, sp_cols)], (sp_rows, sp_cols)),
                          shape=(vec_length, _GEOVECTORIZER_VEC_LENGTH),
                          dtype=np.float32)


        data.append(a)
        vec_length_map.append(vec_length)
        node_idx[m] = i
        m += 1

    logger.debug("Generated {} unique wktLiteral features".format(m))

    if m <= 0:
        return None

    # normalization
    for i in (0, 1):  # only on coordinates
        values = [v for a in data for v in a.data[np.where(a.col==i)]]
        value_max = max(values)
        value_min = min(values)

        for a in data:
            idx = np.where(a.col==i)
            a.data[idx] = (2*(a.data[idx] - value_min) /
                           (value_max - value_min)) - 1.0

    return [[data, node_idx[:m], C, vec_length_map]]

def validate(value):
    return fullmatch(_REGEX_WKTLITERAL, value)
