#!/usr/bin/python3

import logging
from re import fullmatch, sub
from string import punctuation

import numpy as np
from nltk.stem import PorterStemmer
from rdflib.term import Literal
from rdflib.namespace import XSD


_REGEX_CHAR = "[\u0001-\uD7FF\uE000-\uFFFD\u10000-\u10FFFF]"
_REGEX_STRING = "{}*".format(_REGEX_CHAR)
_MAX_CHARS = 512  # ASCII encoded

logger = logging.getLogger(__name__)

def generate_features(nodes_map, node_predicate_map, config):
    """ Generate features for XSD string literals

    Definition
    - string := Char*
    -- Char  := [\t\r\n] + {unicode} + {ISO/IEC 10646}

    This is a character level encoding, in which each character is represented
    by its ASCII value. Prior to encoding, all words were stemmed and
    punctuation was filtered.
    Note that this encoding treats similar spelling as roughly the same word

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
    m = 0
    n = len(nodes_map)
    sequences = np.zeros(shape=(n, _MAX_CHARS), dtype=np.float32)
    node_idx = np.zeros(shape=(n), dtype=np.int32)
    seq_length_map = list()

    value_max = None
    value_min = None

    stemmer = PorterStemmer()  # English
    for node, i in nodes_map.items():
        if type(node) is not Literal:
            continue
        if node.datatype is None or node.datatype.neq(XSD.string):
            continue

        node._value = node.__str__()  ## empty value bug workaround
        seq = validate(node.value)
        if seq is None:  # if invalid syntax
            continue

        seq = preprocess(seq, stemmer)
        vec = np.array([ord(c) for c in seq], dtype=np.float32)
        vec_length = len(vec)
        seq_length_map.append(vec_length)

        if value_max is None or vec.max() > value_max:
            value_max = vec.max()
        if value_min is None or vec.min() < value_min:
            value_min = vec.min()

        # pad with repetition
        sequences[m] = np.append(vec, vec[:_MAX_CHARS-vec_length])[:_MAX_CHARS]
        node_idx[m] = i
        m += 1

    logger.debug("Generated {} unique string features".format(m))

    # normalization over features
    sequences = (sequences[:m] - value_min) / (value_max - value_min)

    return [sequences[:m], node_idx[:m], C, seq_length_map]

def preprocess(seq, stemmer):
    seq = seq.lower()
    seq = sub('['+punctuation+']', '', seq).split()

    for i in range(len(seq)):
        seq[i] = stemmer.stem(seq[i])

    return " ".join(seq)

def validate(value):
    return fullmatch(_REGEX_STRING, value)
