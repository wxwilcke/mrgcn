#!/usr/bin/python3

import logging
from re import fullmatch, sub
from string import punctuation

import numpy as np
from rdflib.term import Literal
from rdflib.namespace import XSD
import scipy.sparse as sp


_REGEX_CHAR = "[\u0001-\uD7FF\uE000-\uFFFD\u10000-\u10FFFF]"
_REGEX_STRING = "{}+".format(_REGEX_CHAR)  # skip empty string
_MAX_CHARS = 512  # one-hot encoded
_VOCAB = [chr(32)] + [chr(i) for i in range(97, 123)] + [chr(i) for i in range(48, 58)]
_VOCAB_MAP = {v:k for k,v in enumerate(_VOCAB)}
_VOCAB_MAX_IDX = len(_VOCAB)

logger = logging.getLogger(__name__)

def generate_features(nodes_map, node_predicate_map, config):
    """ Generate features for XSD string literals

    Definition
    - string := Char*
    -- Char  := [\t\r\n] + {unicode} + {ISO/IEC 10646}

    This is a character level encoding, in which each character is represented
    by a one-hot vector. Prior to encoding, all words
    are stemmed and punctuation is filtered.
    Note that this encoding treats similar spelling as roughly the same word
    Note that UNICODE is ignored as it skews the value distribution

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param config: configuration dictionary
    :returns: numpy array M x A x C;
                    M :- number of nodes with this feature, such that M <= N
                    A :- number of allowed characters
                    C :- maximum length of sequences
              numpy array 1 x M;
                    M :- number of nodes with this feature, such that M <= N
              int C;
              list 1 x M;
              """
    logger.debug("Generating string features")

    C = 16

    if True:
        return generate_relationwise_features(nodes_map, node_predicate_map, C, config)
    else:
        return generate_nodewise_features(nodes_map, C, config)

def generate_nodewise_features(nodes_map, C, config):
    """ Stack all vectors without regard of their relation
    """
    m = 0
    n = len(nodes_map)
    node_idx = np.zeros(shape=(n), dtype=np.int32)
    seq_length_map = list()

    data = list()

    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype.neq(XSD.string):
            continue

        node._value = str(node)  ## empty value bug workaround
        #if validate(node.value) is None:  # if invalid syntax
        #    continue

        sequence = preprocess(node.value)
        sequence = encode(sequence)[:_MAX_CHARS]
        seq_length = len(sequence)
        if seq_length <= 0:
            continue

        a = sp.coo_matrix((np.repeat([1.0], repeats=seq_length),
                           (sequence, np.array(range(seq_length)))),
                          shape=(_VOCAB_MAX_IDX, seq_length),
                          dtype=np.float32)

        data.append(a)
        seq_length_map.append(seq_length)
        node_idx[m] = i
        m += 1

    logger.debug("Generated {} unique string features".format(m))

    if m <= 0:
        return None

    return [[data, node_idx[:m], C, seq_length_map, 1]]

def generate_relationwise_features(nodes_map, node_predicate_map, C, config):
    """ Stack vectors row-wise per relation and column stack relations
    """
    n = len(node_predicate_map)
    m = dict()
    node_idx = dict()
    sequences = dict()
    seq_length_map = dict()
    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype.neq(XSD.string):
            continue

        node._value = str(node)  ## empty value bug workaround
        #if validate(node.value) is None:  # if invalid syntax
        #    continue

        sequence = preprocess(node.value)
        sequence = encode(sequence)[:_MAX_CHARS]
        seq_length = len(sequence)

        if seq_length <= 0:
            continue

        predicate = node_predicate_map[node]
        if predicate not in sequences.keys():
            sequences[predicate] = list()
            m[predicate] = 0
            seq_length_map[predicate] = list()
            node_idx[predicate] = np.zeros(shape=(n), dtype=np.int32)

        a = sp.coo_matrix((np.repeat([1.0], repeats=seq_length),
                           (sequence, np.array(range(seq_length)))),
                          shape=(_VOCAB_MAX_IDX, seq_length),
                          dtype=np.float32)

        sequences[predicate].append(a)

        seq_length_map[predicate].append(seq_length)
        node_idx[predicate][m[predicate]] = i
        m[predicate] += 1

    logger.debug("Generated {} unique string features".format(sum(m.values())))

    if len(m) <= 0:
        return None

    npreds = len(sequences.keys())
    return [[sequences[predicate], node_idx[predicate][:m[predicate]], C, seq_length_map[predicate], npreds]
            for predicate in sequences.keys()]

def encode(seq):
    encoding = list()
    for char in seq:
        if char in _VOCAB:
            encoding.append(_VOCAB_MAP[char])

    return encoding

def preprocess(seq):
    seq = seq.lower()
    seq = sub('['+punctuation+']', '', seq).split()

    return " ".join(seq)

def validate(value):
    return fullmatch(_REGEX_STRING, value)
