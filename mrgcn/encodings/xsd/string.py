#!/usr/bin/python3

import logging
from math import ceil
from re import fullmatch, sub
from string import punctuation

import numpy as np
from nltk.stem import PorterStemmer
from rdflib.term import Literal
from rdflib.namespace import XSD


_REGEX_CHAR = "[\u0001-\uD7FF\uE000-\uFFFD\u10000-\u10FFFF]"
_REGEX_STRING = "{}+".format(_REGEX_CHAR)  # skip empty string
_MAX_CHARS = 512  # ASCII encoded

logger = logging.getLogger(__name__)

def generate_features(nodes_map, node_predicate_map, config,
                      separated_domains=False):
    """ Generate features for XSD string literals

    Definition
    - string := Char*
    -- Char  := [\t\r\n] + {unicode} + {ISO/IEC 10646}

    This is a character level encoding, in which each character is represented
    by its ASCII value. Prior to encoding, all words were stemmed and
    punctuation was filtered.
    Note that this encoding treats similar spelling as roughly the same word
    Note that UNICODE is ignored as it skews the value distribution

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param config: configuration dictionary
    :returns: numpy array M x C;
                    M :- number of nodes with this feature, such that M <= N
                    C :- number of columns for this feature encoding
              numpy array 1 x M;
                    M :- number of nodes with this feature, such that M <= N
              int C;
              None;
              """
    logger.debug("Generating string features")

    C = 128

    if separated_domains:
        return generate_relationwise_features(nodes_map, node_predicate_map, C, config)
    else:
        return generate_nodewise_features(nodes_map, C, config)

def generate_nodewise_features(nodes_map, C, config):
    """ Stack all vectors without regard of their relation
    """
    m = 0
    n = len(nodes_map)
    sequences = np.zeros(shape=(n, _MAX_CHARS), dtype=np.float32)
    node_idx = np.zeros(shape=(n), dtype=np.int32)
    seq_length_map = list()

    value_max = None
    value_min = None

    stemmer = PorterStemmer()  # English
    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype.neq(XSD.string):
            continue

        node._value = str(node)  ## empty value bug workaround
        if validate(node.value) is None:  # if invalid syntax
            continue

        seq = preprocess(node.value, stemmer)
        vec = np.array(toASCII(seq), dtype=np.float32)
        vec_length = len(vec)

        if vec_length <= 0:
            continue
        seq_length_map.append(vec_length)

        if value_max is None or vec.max() > value_max:
            value_max = vec.max()
        if value_min is None or vec.min() < value_min:
            value_min = vec.min()

        # pad with repetition
        sequences[m] = np.tile(vec, ceil(_MAX_CHARS/vec_length))[:_MAX_CHARS]
        node_idx[m] = i
        m += 1

    logger.debug("Generated {} unique string features".format(m))

    # normalization over features
    sequences = 2*(sequences[:m] - value_min) / (value_max - value_min) - 1.0

    return [sequences[:m], node_idx[:m], C, seq_length_map]

def generate_relationwise_features(nodes_map, node_predicate_map, C, config):
    """ Stack vectors row-wise per relation and column stack relations
    """
    m = 0
    n = len(node_predicate_map)
    relationwise_encodings = dict()
    node_idx = np.zeros(shape=(n), dtype=np.int32)
    seq_length_map = list()
    values_idx = dict()
    values_min = dict()
    values_max = dict()
    stemmer = PorterStemmer()  # English
    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype.neq(XSD.string):
            continue

        node._value = str(node)  ## empty value bug workaround
        if validate(node.value) is None:  # if invalid syntax
            continue

        seq = preprocess(node.value, stemmer)
        vec = np.array(toASCII(seq), dtype=np.float32)
        vec_length = len(vec)

        if vec_length <= 0:
            continue
        seq_length_map.append(vec_length)

        predicate = node_predicate_map[node]
        if predicate not in relationwise_encodings.keys():
            relationwise_encodings[predicate] = np.zeros(shape=(n, _MAX_CHARS), dtype=np.float32)
            values_min[predicate] = None
            values_max[predicate] = None
            values_idx[predicate] = list()

        if values_max[predicate] is None or vec.max() > values_max[predicate]:
            values_max[predicate] = vec.max()
        if values_min[predicate] is None or vec.min() < values_min[predicate]:
            values_min[predicate] = vec.min()


        # pad with repetition
        relationwise_encodings[predicate][m] = np.tile(vec, ceil(_MAX_CHARS/vec_length))[:_MAX_CHARS]
        node_idx[m] = i
        values_idx[predicate].append(m)
        m += 1

    logger.debug("Generated {} unique string features".format(m))

    # normalization over encodings
    for predicate, encodings in relationwise_encodings.items():
        encodings[values_idx[predicate]] = (2*(encodings[values_idx[predicate]] - values_min[predicate]) /
                                             (values_max[predicate] - values_min[predicate])) -1.0

    encodings = np.hstack([encodings[:m] for encodings in
                           relationwise_encodings.values()])

    return [encodings[:m], node_idx[:m], C, seq_length_map]

def toASCII(seq):
    try:
        return [c for c in seq.encode('ascii')]
    except UnicodeEncodeError:
        return []

def preprocess(seq, stemmer):
    seq = seq.lower()
    seq = sub('['+punctuation+']', '', seq).split()

    for i in range(len(seq)):
        seq[i] = stemmer.stem(seq[i])

    return " ".join(seq)

def validate(value):
    return fullmatch(_REGEX_STRING, value)
