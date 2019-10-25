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
_MAX_CHARS = 128

logger = logging.getLogger(__name__)

def generate_features(nodes_map, config):
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
    :returns: list with B numpy arrays Nb x Cb;
                    Nb :- number of encoded strings for nodes in batch b
                    Cb :- number of columns for batch b
              list with B lists of length Nb, such that B[i][j] holds node idx of
              encoding j in batch i
              """
    logger.debug("Generating string features")

    m = 0
    n = len(nodes_map)
    sequences = np.zeros(shape=(n, _MAX_CHARS), dtype=np.float32)
    node_idx = np.zeros(shape=(n), dtype=np.int32)
    seq_length_map = list()

    value_max = -1
    value_min = -1

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

        if vec.max() > value_max or value_max == -1:
            value_max = vec.max()
        if vec.min() < value_min or value_min == -1:
            value_min = vec.min()

        # pad with repetition
        sequences[m] = np.append(vec, vec[:_MAX_CHARS-vec_length])[:_MAX_CHARS]
        node_idx[m] = i
        m += 1

    logger.debug("Generated {} unique string features".format(m))

    # inplace L1 normalization over features
    if config['normalize']:
        sequences = (sequences - value_min) / (value_max - value_max -
                                               value_min)

    bins, bin_assignments = subdivide_into_bins(sequences, node_idx, seq_length_map)

    return [bins, bin_assignments]

def subdivide_into_bins(sequences, node_idx, seq_length_map):
    # determine optimal number of bins using the Freedman-Diaconis rule
    IQR = np.quantile(seq_length_map, 0.75) - np.quantile(seq_length_map, 0.25)
    h = 2 * IQR / np.power(len(seq_length_map), 1/3)
    nbins = np.round((max(seq_length_map)-min(seq_length_map)) / h)

    bin_ranges = np.array_split(np.unique(seq_length_map), nbins)
    bin_ranges_map = {length:bin_idx for bin_idx in range(len(bin_ranges))
                      for length in bin_ranges[bin_idx]}
    bin_assignments = [list() for bin_range in bin_ranges]
    for i in range(len(sequences)):
        length = seq_length_map[i]
        bin_assignments[bin_ranges_map[length]].append(i)

    bins = [np.array(sequences[idc, :max(ranges)])
            for idc, ranges in zip(bin_assignments, bin_ranges)]

    return [bins, bin_assignments]

def preprocess(seq, stemmer):
    seq = seq.lower()
    seq = sub('['+punctuation+']', '', seq).split()

    for i in range(len(seq)):
        seq[i] = stemmer.stem(seq[i])

    return " ".join(seq)

def validate(value):
    return fullmatch(_REGEX_STRING, value)
