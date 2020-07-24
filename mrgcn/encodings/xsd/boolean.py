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

    Definition
    - boolean := 'true' | 'false' | '1' | '0'

    Numerical booleans and their text equivalents are mapped to [-1, 1] for
    maximum separation in the value space. No normalization is needed.

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param node_predicate_map: dictionary of node labels (URIs): {predicates}
    :param config: configuration dictionary
    :returns: list of length P with lists Q of length 3;
                P :- number of predicates that link to nodes with this feature
                Q :- [enc, node_idx, None];
                    enc :- numpy array M x C;
                        M :- number of nodes with this feature, such that M <= N
                    node_idx :- numpy vector of length M, mapping seq index to node id
                    None :- not used here
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
        #if validate(value) is None:
        #    continue

        b = 1.0 if value == "true" or value == "1" else -1.0

        # add to matrix structures
        encodings[m] = [b]
        node_idx[m] = i
        m += 1

    logger.debug("Generated {} unique boolean encodings".format(m))

    if m <= 0:
        return None

    return [[encodings[:m], node_idx[:m], None]]

def generate_relationwise_features(nodes_map, node_predicate_map, C, config):
    n = len(nodes_map)
    m = dict()
    relationwise_encodings = dict()
    node_idx = dict()

    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype.neq(XSD.boolean):
            continue

        value = str(node)  ## empty value bug workaround
        #if validate(value) is None:
        #    continue

        b = 1.0 if value == "true" or value == "1" else -1.0

        for predicate in node_predicate_map[node]:
            if predicate not in relationwise_encodings.keys():
                relationwise_encodings[predicate] = np.zeros(shape=(n, C), dtype=np.float32)
                node_idx[predicate] = np.zeros(shape=(n), dtype=np.int32)
                m[predicate] = 0

            # add to matrix structures
            relationwise_encodings[predicate][m[predicate]] = [b]
            node_idx[predicate][m[predicate]] = i
            m[predicate] += 1

    logger.debug("Generated {} unique boolean encodings".format(
        sum(m.values())))

    if len(m) <= 0:
        return None

    return [[encodings[:m[pred]], node_idx[pred][:m[pred]], None]
            for pred, encodings in relationwise_encodings.items()]

def validate(value):
    return match(_REGEX_BOOLEAN, value)
