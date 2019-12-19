#!/usr/bin/python3

import logging
from re import match

import numpy as np
from rdflib.term import Literal
from rdflib.namespace import XSD


_REGEX_BOOLEAN = "true|false|0|1"

logger = logging.getLogger(__name__)

def generate_features(nodes_map, node_predicate_map, config):
    """ Generate encodings for XSD boolean literals

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param config: configuration dictionary
    :returns: ...
    """
    logger.debug("Generating boolean encodings")
    C = 1  # number of items per feature

    if True:  #not config['share_weights']:
        return generate_relationwise_features(nodes_map, node_predicate_map, C,
                                              config)
    else:
        return generate_nodewise_features(nodes_map, C, config)


def generate_nodewise_features(nodes_map, C, config):
    """ Stack all vectors without regard of their relation
    """
    m = 0
    n = len(nodes_map)
    encodings = np.zeros(shape=(n, C), dtype=np.float32)
    node_idx = np.zeros(shape=(n), dtype=np.int32)
    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype.neq(XSD.boolean):
            continue

        value = str(node)  ## empty value bug workaround
        if validate(value) is None:
            continue

        b = 1.0 if value == "true" or value == "1" else -1.0

        # add to matrix structures
        encodings[m] = [b]
        node_idx[m] = i
        m += 1

    logger.debug("Generated {} unique boolean encodings".format(m))

    if m <= 0:
        return None

    return [[encodings[:m], node_idx[:m], C, None]]

def generate_relationwise_features(nodes_map, node_predicate_map, C, config):
    n = len(nodes_map)
    m = 0
    relationwise_encodings = dict()
    node_idx = np.zeros(shape=(n), dtype=np.int32)

    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype.neq(XSD.boolean):
            continue

        value = str(node)  ## empty value bug workaround
        if validate(value) is None:
            continue

        b = 1.0 if value == "true" or value == "1" else -1.0

        predicate = node_predicate_map[node]
        if predicate not in relationwise_encodings.keys():
            relationwise_encodings[predicate] = np.zeros(shape=(n, C), dtype=np.float32)

        # add to matrix structures
        relationwise_encodings[predicate][m] = [b]
        node_idx[m] = i
        m += 1

    logger.debug("Generated {} unique boolean encodings".format(m))

    if m <= 0:
        return None

    encodings = np.hstack([encodings[:m] for encodings in
                           relationwise_encodings.values()])
    C *= len(relationwise_encodings.keys())

    return [[encodings[:m], node_idx[:m], C, None]]

def validate(value):
    return match(_REGEX_BOOLEAN, value)
