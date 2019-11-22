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
_MAX_CHARS = 64  # one-hot ASCII encoded
_MAX_ASCII = 255

logger = logging.getLogger(__name__)

def generate_features(nodes_map, node_predicate_map, config,
                      separated_domains=False):
    """ Generate features for XSD string literals

    Definition
    - string := Char*
    -- Char  := [\t\r\n] + {unicode} + {ISO/IEC 10646}

    This is a character level encoding, in which each character is represented
    by a one-hot vector indexed by its ASCII value. Prior to encoding, all words
    are stemmed and punctuation is filtered.
    Note that this encoding treats similar spelling as roughly the same word
    Note that UNICODE is ignored as it skews the value distribution

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param config: configuration dictionary
    :returns: numpy array M x A x C;
                    M :- number of nodes with this feature, such that M <= N
                    A :- number of allowed ASCII characters
                    C :- maximum length of sequences
              numpy array 1 x M;
                    M :- number of nodes with this feature, such that M <= N
              int C;
              list 1 x M;
              """
    logger.debug("Generating string features")

    C = 32

    if separated_domains:
        return generate_relationwise_features(nodes_map, node_predicate_map, C, config)
    else:
        return generate_nodewise_features(nodes_map, C, config)

def generate_nodewise_features(nodes_map, C, config):
    """ Stack all vectors without regard of their relation
    """
    m = 0
    n = len(nodes_map)
    sequences = list()
    node_idx = np.zeros(shape=(n), dtype=np.int32)
    seq_length_map = list()

    min_value = None
    max_value = None

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
        seq = toASCII(seq)[:_MAX_CHARS]
        seq_length = len(seq)
        if seq_length <= 0:
            continue

        sequence = np.zeros(shape=(_MAX_ASCII, _MAX_CHARS), dtype=np.float32)
        sequence[[seq], range(seq_length)] = 1.0  # not sparse as Conv1D doesn't support it

        # pad with repetition
        unfilled = _MAX_CHARS - seq_length
        if unfilled > 0:
            sequence[:,seq_length:] = np.tile(sequence[:,:seq_length],
                                              ceil((unfilled)/seq_length))[:,:unfilled]
        sequences.append(sequence)

        if min_value is None or min(seq) < min_value:
            min_value = min(seq)
        if max_value is None or max(seq) > max_value:
            max_value = max(seq)

        seq_length_map.append(seq_length)
        node_idx[m] = i
        m += 1

    logger.debug("Generated {} unique string features".format(m))

    # remove unused ASCII range as we're not interested in unseen data right now
    sequences = np.stack(sequences, axis=0)[:,min_value:max_value,:max(seq_length_map)]

    return [[sequences, node_idx[:m], C, seq_length_map]]

def generate_relationwise_features(nodes_map, node_predicate_map, C, config):
    """ Stack vectors row-wise per relation and column stack relations
    """
    n = len(node_predicate_map)
    m = dict()
    node_idx = dict()
    sequences = dict()
    seq_length_map = dict()
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
        seq = toASCII(seq)[:_MAX_CHARS]
        seq_length = len(seq)

        if seq_length <= 0:
            continue

        predicate = node_predicate_map[node]
        if predicate not in sequences.keys():
            sequences[predicate] = list()
            m[predicate] = 0
            seq_length_map[predicate] = list()
            node_idx[predicate] = np.zeros(shape=(n), dtype=np.int32)
            values_min[predicate] = None
            values_max[predicate] = None

        sequence = np.zeros(shape=(_MAX_ASCII, _MAX_CHARS), dtype=np.float32)
        sequence[[seq], range(seq_length)] = 1.0  # not sparse as Conv1D doesn't support it

        # pad with repetition
        unfilled = _MAX_CHARS - seq_length
        if unfilled > 0:
            sequence[:,seq_length:] = np.tile(sequence[:,:seq_length],
                                              ceil((unfilled)/seq_length))[:,:unfilled]
        sequences[predicate].append(sequence)

        if values_max[predicate] is None or max(seq) > values_max[predicate]:
            values_max[predicate] = max(seq)
        if values_min[predicate] is None or min(seq) < values_min[predicate]:
            values_min[predicate] = min(seq)

        seq_length_map[predicate].append(seq_length)
        node_idx[m[predicate]] = i
        m[predicate] += 1

    logger.debug("Generated {} unique string features".format(sum(m.values())))

    for predicate, sequence in sequences.items():
        # remove unused ASCII range as we're not interested in unseen data right now
        sequence = np.stack(sequence,
                            axis=0)[:,values_min[predicate]:values_max[predicate],:max(seq_length_map[predicate])]

    return [[sequences[predicate], node_idx[predicate][:m[predicate]], C, seq_length_map[predicate]]
            for predicate in sequences.keys()]

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
