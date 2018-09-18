#!/usr/bin/python3

import logging
from re import fullmatch, match

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from rdflib.term import Literal
from rdflib.namespace import XSD


_REGEX_CHAR = "[\u0001-\uD7FF\uE000-\uFFFD\u10000-\u10FFFF]"
_REGEX_STRING = "{}*".format(_REGEX_CHAR)

logger = logging.getLogger(__name__)

def generate_features(nodes_map, config):
    """ Generate features for XSD string literals

    Definition
    - string := Char*
    -- Char  := [\t\r\n] + {unicode} + {ISO/IEC 10646}

    Note: ... 

    Embedding
    - a vector v of length C = ?
    -- v[0] : ... 

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param config: configuration dictionary
    :returns: scipy sparse matrix N x C;
                    N :- number of nodes
                    C :- number of columns for this feature embedding
    """
    logger.debug("Generating string features")
    C = ?  # number of items per feature

    rows = []
    data = []
    for node, i in nodes_map.items():
        if type(node) is not Literal:
            continue
        if node.datatype is None or node.datatype.neq(XSD.string):
            continue

        node._value = node.__str__()  ## empty value bug workaround
        m = validate(node.value)
        if m is None:  # invalid syntax
            continue

        ...

        # add to matrix structures
        rows.append(i)
        data.extend([?])

    cols = np.tile(range(C), len(rows))
    rows = np.repeat(rows, C)  # expand indices

    logger.debug("Generated {} unique string features".format(len(cols)//C))

    # create matrix
    features = sp.csr_matrix((data, (rows, cols)),
                         shape=(len(nodes_map), C),
                         dtype=np.float32)

    # inplace L1 normalization over features
    if config['normalize']:
        features = normalize(features, norm='l1', axis=0)

    return features

def validate(value):
    return fullmatch(_REGEX_STRING, value)
