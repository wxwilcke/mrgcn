#!/usr/bin/python3

import logging
from re import match

import numpy as np
from rdflib.term import Literal
from rdflib.namespace import XSD


_REGEX_YEAR_FRAG = "(?P<sign>-?)(?P<year>\d{4})"  # only consider years from -9999 to 9999
_REGEX_TIMEZONE_FRAG = "(?P<timezone>Z|(?:\+|-)(?:(?:0\d|1[0-3]):[0-5]\d|14:00))?"

logger = logging.getLogger(__name__)

def generate_features(nodes_map, node_predicate_map, config):
    """ Generate encodings for XSD gYear literals

    Definition
    - gYear := yearFrag timezoneFrag?
    -- yearFrag := '-'? (([1-9] \d\d\d+)) | ('0' \d\d\d))
    -- timezoneFrag := 'Z' | ('+' | '-') (('0' \d | '1' [0-3]) ':' minuteFrag | '14:00')
    ---- minuteFrag := [0-5] \d

    Note: a. We consider only yearFrag; timezoneFrag is too fine-grained and only
          makes sense within a time span of days. Consider using xsd:date or
          xsd:dateTime for that purpose.
          b. We consider only years from 9999 BCE to 9999 AD. Hence, we redefine
          yearFrag as '-'? \d{4}

    Returns an 2D array A and an vector b, such that A[i] holds the vector
    representation of the feature belonging to node b[i].

    Encoding
    - a vector v of length C = 4
    -- v[0] : '-'? : BCE or AD; 0.0 if '-', else -1.0
                     Note: a. needed to represent difference 0YY AD and 0YY BCE
                           b. mapping assumes majority is AD
    -- v[1] : \d\d : centuries; numerical and normalized
                     Note: no separation between hundred and thousands as the
                           latter's range is typically limited
    -- v[2] : \d   : decades; numerical and normalized
    -- v[3] : \d   : individual years; numerical and normalized


    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param config: configuration dictionary
    :returns: numpy array M x C;
                    M :- number of nodes with a gYear feature, such that M <= N
                    C :- number of columns for this feature embedding
              numpy array 1 x M;
                    M :- number of nodes with a gYear feature, such that M <= N
    """
    logger.debug("Generating gYear encodings")
    C = 4  # number of items per feature

    m = 0
    n = len(nodes_map)
    encodings = np.zeros(shape=(n, C), dtype=np.float32)
    node_idx = np.zeros(shape=(n), dtype=np.int32)
    values_max = [None, None, None]
    values_min = [None, None, None]
    for node, i in nodes_map.items():
        if type(node) is not Literal:
            continue
        if node.datatype is None or node.datatype.neq(XSD.gYear):
            continue

        node._value = node.__str__()  ## empty value bug workaround
        value = validate(node.value)
        if value is None:  # invalid syntax
            continue

        sign = 0. if value.group('sign') == '' else 1.
        year = value.group('year')

        # separate centuries, decades, and individual years
        separated = separate(year)
        if separated is None:
            continue

        c = int(separated.group('century'))
        d = int(separated.group('decade'))
        y = int(separated.group('year'))

        for i, value in enumerate([c,d,y]):
            if values_max[i] is None or value > values_max[i]:
                values_max[i] = value
            if values_min[i] is None or value < values_min[i]:
                values_min[i] = value

        # add to matrix structures
        encodings[m] = [sign, c, d, y]
        node_idx[m] = i
        m += 1

    logger.debug("Generated {} unique gYear encodings".format(m))

    # normalization over encodings
    for i in range(C-1):
        # skip sign as it can only take values [0, 1]
        encodings[:m,i+1] = ((encodings[:m,i+1] - values_min[i]) /
                             (values_max[i] - values_min[i]))

    return [encodings[:m], node_idx[:m], C, None]

def separate(year):
    regex = "(?P<century>\d\d)(?P<decade>\d)(?P<year>\d)"
    return match(regex, year)

def validate(value):
    return match("{}{}".format(_REGEX_YEAR_FRAG,
                               _REGEX_TIMEZONE_FRAG),
                 value)
