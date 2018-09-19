#!/usr/bin/python3

import logging
from re import match

import numpy as np
from sklearn.preprocessing import normalize
from rdflib.term import Literal
from rdflib.namespace import XSD


_REGEX_YEAR_FRAG = "(?P<sign>-?)(?P<year>\d{4})"  # only consider years from -9999 to 9999
_REGEX_TIMEZONE_FRAG = "(?P<timezone>Z|(?:\+|-)(?:(?:0\d|1[0-3]):[0-5]\d|14:00))?"

logger = logging.getLogger(__name__)

def generate_features(nodes_map, config):
    """ Generate features for XSD gYear literals

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

    Embedding
    - a vector v of length C = 4
    -- v[0] : '-'? : BCE or AD; 1.0 if '-', else 0.0
                     Note: a. needed to represent difference 0YY AD and 0YY BCE
                           b. mapping assumes majority is AD
    -- v[1] : \d\d : centuries; numerical and normalized
                     Note: no separation between hundred and thousands as the
                           latter's range is typically limited
    -- v[2] : \d   : decades; numerical and normalized
    -- v[3] : \d   : individual years; numerical and normalized


    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param config: configuration dictionary
    :returns: numpy array N x C;
                    N :- number of nodes
                    C :- number of columns for this feature embedding
    """
    logger.debug("Generating gYear features")
    C = 4  # number of items per feature

    nfeatures = 0
    features = np.zeros(shape=(len(nodes_map), C), dtype=np.float32)
    for node, i in nodes_map.items():
        if type(node) is not Literal:
            continue
        if node.datatype is None or node.datatype.neq(XSD.gYear):
            continue

        node._value = node.__str__()  ## empty value bug workaround
        m = validate(node.value)
        if m is None:  # invalid syntax
            continue

        sign = 0. if m.group('sign') == '' else 1.
        year = m.group('year')

        # separate centuries, decades, and individual years
        separated = separate(year)
        if separated is None:
            continue

        c = int(separated.group('century'))
        d = int(separated.group('decade'))
        y = int(separated.group('year'))

        # add to matrix structures
        features[i] = [sign, c, d, y]
        nfeatures += 1

    logger.debug("Generated {} unique gYear features".format(nfeatures))

    # inplace L1 normalization over features
    if config['normalize']:
        features = normalize(features, norm='l1', axis=0)

    return features

def separate(year):
    regex = "(?P<century>\d\d)(?P<decade>\d)(?P<year>\d)"
    return match(regex, year)

def validate(value):
    return match("{}{}".format(_REGEX_YEAR_FRAG,
                               _REGEX_TIMEZONE_FRAG),
                 value)
