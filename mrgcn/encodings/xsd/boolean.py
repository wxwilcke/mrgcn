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

    return [[encodings[:m], node_idx[:m], C, None, 1]]

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

        predicate = node_predicate_map[node]
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

    npreds = len(relationwise_encodings.keys())

    return [[encodings[:m[pred]], node_idx[pred][:m[pred]], C, None, npreds]
            for pred, encodings in relationwise_encodings.items()]

def validate(value):
    return match(_REGEX_BOOLEAN, value)
