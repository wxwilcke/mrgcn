#!/usr/bin/python3

import logging
from re import match

import numpy as np
from rdflib.term import Literal
from rdflib.namespace import XSD


_REGEX_NUMERIC = "\d+"

_XSD_NUMERICAL = {
    XSD.double,
    XSD.decimal,
    XSD.float,
    XSD.integer,
    XSD.long,
    XSD.int,
    XSD.short,
    XSD.byte,
    XSD.nonNegativeInteger,
    XSD.nonPositiveInteger,
    XSD.unsignedLong,
    XSD.unsignedInt,
    XSD.unsignedShort,
    XSD.unsignedByte,
    XSD.negativeInteger,
    XSD.positiveInteger}

logger = logging.getLogger(__name__)

def generate_features(node_map, node_predicate_map, config, separated_domains=True):
    """ Generate encodings for XSD numeric literals

    Definition
    - numeric := \d+

    Returns an 2D array A and an vector b, such that A[i] holds the vector
    representation of the feature belonging to node b[i].

    Encoding
    - a vector v of length C = 1
    -- v[0] : \d+ : numerical value(s)

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param config: configuration dictionary
    :returns: numpy array M x C;
                    M :- number of nodes with a gYear feature, such that M <= N
                    C :- number of columns for this feature embedding
              numpy array 1 x M;
                    M :- number of nodes with a gYear feature, such that M <= N
    """
    logger.debug("Generating numerical encodings")
    C = 1  # number of items per feature per relation

    if separated_domains:
        return generate_relationwise_features(node_map, node_predicate_map, C, config)
    else:
        return generate_nodewise_features(node_map, C, config)

def generate_nodewise_features(node_map, C, config):
    """ Stack all vectors without regard of their relation
    """
    m = 0
    n = len(node_map)
    encodings = np.zeros(shape=(n, C), dtype=np.float32)
    node_idx = np.zeros(shape=(n), dtype=np.int32)
    value_max = None
    value_min = None
    for node, i in node_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype not in _XSD_NUMERICAL:
            continue

        node._value = node.__str__()  ## empty value bug workaround
        value = validate(node.value)
        if value is None:  # invalid syntax
            continue

        if value_max is None or value > value_max:
            value_max = value
        if value_min is None or value < value_min:
            value_min = value

        # add to matrix structures
        encodings[m] = [value]
        node_idx[m] = i
        m += 1

    logger.debug("Generated {} unique {} encodings".format(m, node.datatype))

    # normalization over encodings
    encodings[:m] = (2*(encodings[:m] - value_min) /
                     (value_max - value_min)) -1.0

    return [[encodings[:m], node_idx[:m], C, None]]

def generate_relationwise_features(node_map, node_predicate_map, C, config):
    """ Stack vectors row-wise per relation and column stack relations
    """
    n = len(node_predicate_map)
    m = 0
    relationwise_encodings = dict()
    node_idx = np.zeros(shape=(n), dtype=np.int32)
    values_idx = dict()
    values_min = dict()
    values_max = dict()
    for node, i in node_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype not in _XSD_NUMERICAL:
            continue

        node._value = node.__str__()  ## empty value bug workaround
        value = validate(node.value)
        if value is None:  # invalid syntax
            continue

        predicate = node_predicate_map[node]
        if predicate not in relationwise_encodings.keys():
            relationwise_encodings[predicate] = np.zeros(shape=(n, C), dtype=np.float32)
            values_min[predicate] = None
            values_max[predicate] = None
            values_idx[predicate] = list()

        if values_max[predicate] is None or value > values_max[predicate]:
            values_max[predicate] = value
        if values_min[predicate] is None or value < values_min[predicate]:
            values_min[predicate] = value

        # add to matrix structures
        relationwise_encodings[predicate][m] = [value]
        node_idx[m] = i
        values_idx[predicate].append(m)
        m += 1

    logger.debug("Generated {} unique {} encodings".format(m, node.datatype))

    # normalization over encodings
    for predicate, encodings in relationwise_encodings.items():
        encodings[values_idx[predicate]] = (2*(encodings[values_idx[predicate]] - values_min[predicate]) /
                                             (values_max[predicate] - values_min[predicate])) -1.0

    encodings = np.hstack([encodings[:m] for encodings in
                           relationwise_encodings.values()])
    C *= len(relationwise_encodings.keys())

    return [[encodings[:m], node_idx[:m], C, None]]

def validate(value):
    return match(_REGEX_NUMERIC, value)
