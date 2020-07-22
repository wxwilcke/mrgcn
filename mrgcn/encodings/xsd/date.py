#!/usr/bin/python3

import logging
from math import pi, sin, cos
from re import match

import numpy as np
from rdflib.term import Literal
from rdflib.namespace import XSD


_REGEX_YEAR_FRAG = "(?P<sign>-?)(?P<year>\d{4})"  # only consider years from -9999 to 9999
_REGEX_MONTH_FRAG = "(?P<month>\d{2})"
_REGEX_DAY_FRAG = "(?P<day>\d{2})"
_REGEX_TIMEZONE_FRAG = "(?P<timezone>Z|(?:\+|-)(?:(?:0\d|1[0-3]):[0-5]\d|14:00))?"
_REGEX_DATE = "{}-{}-{}(?:{})?".format(_REGEX_YEAR_FRAG,
                                     _REGEX_MONTH_FRAG,
                                     _REGEX_DAY_FRAG,
                                     _REGEX_TIMEZONE_FRAG)
_DAY_RAD = 2*pi/31
_MONTH_RAD = 2*pi/12
_YEAR_DECADE_RAD = 2*pi/10

logger = logging.getLogger(__name__)

def generate_features(nodes_map, node_predicate_map, config):
    """ Generate encodings for XSD date literals

    Definition
    - date := yearFrag '-' monthFrag '-' dayFrag timezoneFrag?   

    Encoding
    - a vector v of length C = 4
    -- v[0] : '-'? : BCE or AD; -1.0 if '-', else 1.0
                     Note: a. needed to represent difference 0YY AD and 0YY BCE
                           b. mapping assumes majority is AD
    -- v[1] : \d\d : centuries; numerical and normalized
                     Note: no separation between hundred and thousands as the
                           latter's range is typically limited
    -- v[2:4] : \d,\d   : decades on circle
    -- v[4:6] : \d,\d   : individual years on circle
    -- v[6:8]: \d,\d : point on circle representing month
    -- v[8:10]: \d,\d : point on circle representing day

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param node_predicate_map: dictionary of node labels (URIs): {predicates}
    :param config: configuration dictionary
    :returns: list of length P with lists Q of length 4;
                P :- number of predicates that link to nodes with this feature
                Q :- [enc, node_idx, C, None];
                    enc :- numpy array M x C;
                        M :- number of nodes with this feature, such that M <= N
                    node_idx :- numpy vector of length M, mapping seq index to node id
                    C :- desired output dimension of encoder
                    None :- not used here

    """
    logger.debug("Generating date encodings")
    C = 10  # number of items per feature

    if True:
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
    value_max = None
    value_min = None
    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype.neq(XSD.date):
            continue

        node._value = str(node)  ## empty value bug workaround
        value = validate(node.value)
        if value is None:  # invalid syntax
            continue

        sign = 1. if value.group('sign') == '' else -1.
        year = value.group('year')

        # separate centuries, decades, and individual years
        separated = separate(year)
        if separated is None:
            continue

        decade = int(separated.group('decade'))
        dec1, dec2 = point(decade, _YEAR_DECADE_RAD)

        year = int(separated.group('year'))
        y1, y2 = point(year, _YEAR_DECADE_RAD)

        month = value.group('month')
        m1, m2 = point(int(month), _MONTH_RAD)

        day = value.group('day')
        d1, d2 = point(int(day), _DAY_RAD)

        c = int(separated.group('century'))
        if value_max is None or c > value_max:
            value_max = c
        if value_min is None or c < value_min:
            value_min = c

        # add to matrix structures
        encodings[m] = [sign, c, dec1, dec2, y1, y2, m1, m2, d1, d2]
        node_idx[m] = i
        m += 1

    logger.debug("Generated {} unique date encodings".format(m))

    if m <= 0:
        return None

    # normalization over centuries
    encodings[:m,1] = (2*(encodings[:m,1] - value_min) /
                        (value_max - value_min)) - 1.0

    return [[encodings[:m], node_idx[:m], C, None, 1]]

def generate_relationwise_features(nodes_map, node_predicate_map, C, config):
    """ Stack vectors row-wise per relation and column stack relations
    """
    n = len(nodes_map)
    m = dict()
    relationwise_encodings = dict()
    node_idx = dict()
    values_min = dict()
    values_max = dict()

    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype.neq(XSD.date):
            continue

        node._value = str(node)  ## empty value bug workaround
        value = validate(node.value)
        if value is None:  # invalid syntax
            continue

        sign = 1. if value.group('sign') == '' else -1.
        year = value.group('year')

        # separate centuries, decades, and individual years
        separated = separate(year)
        if separated is None:
            continue

        decade = int(separated.group('decade'))
        dec1, dec2 = point(decade, _YEAR_DECADE_RAD)

        year = int(separated.group('year'))
        y1, y2 = point(year, _YEAR_DECADE_RAD)

        month = value.group('month')
        m1, m2 = point(int(month), _MONTH_RAD)

        day = value.group('day')
        d1, d2 = point(int(day), _DAY_RAD)

        for predicate in node_predicate_map[node]:
            if predicate not in relationwise_encodings.keys():
                relationwise_encodings[predicate] = np.zeros(shape=(n, C),
                                                             dtype=np.float32)
                node_idx[predicate] = np.zeros(shape=(n), dtype=np.int32)
                m[predicate] = 0
                values_min[predicate] = None
                values_max[predicate] = None

            c = int(separated.group('century'))
            if values_max[predicate] is None or c > values_max[predicate]:
                values_max[predicate] = c
            if values_min[predicate] is None or c < values_min[predicate]:
                values_min[predicate] = c

            # add to matrix structures
            relationwise_encodings[predicate][m[predicate]] =\
                    [sign, c, dec1, dec2, y1, y2, m1, m2, d1, d2]
            node_idx[m[predicate]] = i
            m[predicate] += 1

    logger.debug("Generated {} unique date encodings".format(
        sum(m.values())))

    if len(m) <= 0:
        return None

    # normalization over centuries
    for pred in relationwise_encodings.keys():
        idx = np.arange(m[pred])
        if values_max[pred] == values_min[pred]:
            relationwise_encodings[pred][idx,1] = 0.0
            continue

        relationwise_encodings[pred][idx,1] =\
                (2*(relationwise_encodings[pred][idx,1] - values_min[pred]) /
                         (values_max[pred] - values_min[pred])) -1.0

    npreds = len(relationwise_encodings.keys())

    return [[encodings[:m[pred]], node_idx[pred][:m[pred]], C, None, npreds]
            for pred, encodings in relationwise_encodings.items()]

def point(m, rad):
    # place on circle
    return (sin(m*rad), cos(m*rad))

def separate(year):
    regex = "(?P<century>\d\d)(?P<decade>\d)(?P<year>\d)"
    return match(regex, year)

def validate(value):
    return match(_REGEX_DATE, value)
