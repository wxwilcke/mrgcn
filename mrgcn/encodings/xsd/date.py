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

logger = logging.getLogger(__name__)

def generate_features(nodes_map, node_predicate_map, config):
    """ Generate encodings for XSD date literals


    Encoding
    - a vector v of length C = 4
    -- v[0] : '-'? : BCE or AD; -1.0 if '-', else 1.0
                     Note: a. needed to represent difference 0YY AD and 0YY BCE
                           b. mapping assumes majority is AD
    -- v[1] : \d\d : centuries; numerical and normalized
                     Note: no separation between hundred and thousands as the
                           latter's range is typically limited
    -- v[2] : \d   : decades; numerical and normalized
    -- v[3] : \d   : individual years; numerical and normalized
    -- v[4:6]: \d,\d : point on circle representing month
    -- v[6:8]: \d,\d : point on circle representing day


    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param config: configuration dictionary
    :returns: ...
    """
    logger.debug("Generating date encodings")
    C = 8  # number of items per feature

    m = 0
    n = len(nodes_map)
    encodings = np.zeros(shape=(n, C), dtype=np.float32)
    node_idx = np.zeros(shape=(n), dtype=np.int32)
    values_max = [None, None, None]
    values_min = [None, None, None]
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

        c = int(separated.group('century'))
        d = int(separated.group('decade'))
        y = int(separated.group('year'))

        month = value.group('month')
        m1, m2 = point(int(month), _MONTH_RAD)

        day = value.group('day')
        d1, d2 = point(int(day), _DAY_RAD)

        for idx, value in enumerate([c,d,y]):
            if values_max[idx] is None or value > values_max[idx]:
                values_max[idx] = value
            if values_min[idx] is None or value < values_min[idx]:
                values_min[idx] = value

        # add to matrix structures
        encodings[m] = [sign, c, d, y, m1, m2, d1, d2]
        node_idx[m] = i
        m += 1

    logger.debug("Generated {} unique date encodings".format(m))

    # normalization over encodings
    for i in range(3):
        # skip sign as it can only take values [-1, 1]
        encodings[:m,i+1] = (2*(encodings[:m,i+1] - values_min[i]) /
                            (values_max[i] - values_min[i])) - 1.0

    return [[encodings[:m], node_idx[:m], C, None]]

def point(m, rad):
    # place on circle
    return (sin(m*rad), cos(m*rad))

def separate(year):
    regex = "(?P<century>\d\d)(?P<decade>\d)(?P<year>\d)"
    return match(regex, year)

def validate(value):
    return match(_REGEX_DATE, value)
