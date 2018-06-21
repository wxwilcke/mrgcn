#!/usr/bin/python3

import logging
from re import match

import numpy as np
import scipy.sparse as sp
from rdflib.term import Literal
from rdflib.namespace import XSD


_REGEX_YEAR_FRAG = "(?P<sign>-?)(?P<year>\d{4})"  # only consider years from -9999 to 9999
_REGEX_TIMEZONE_FRAG = "(?P<timezone>Z|(?:\+|-)(?:(?:0\d|1[0-3]):[0-5]\d|14:00))?"

logger = logging.getLogger(__name__)

def generate_features(nodes_map):
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
          c. All digits are represented as one-hot vectors, as they are more
          akin to intervals

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
    :returns: scipy sparse matrix N x C; 
                    N :- number of nodes
                    C :- number of columns for this feature embedding
    """
    C = 4  # number of items per feature 

    rows = []
    data = []
    for node, i in nodes_map:
        if type(node) is not Literal:
            continue
        if node.datatype is not XSD.gYear:
            continue

        m = validate(node.value)
        if m is None:  # invalid syntax
            continue

        sign = m.group('sign')
        year = m.group('year')

        # separate centuries, decades, and individual years
        c, d, y = separate(year)

        # add to matrix structures
        rows.append(i)
        data.extend([sign, c, d, y])
    
    rows = np.repeat(rows, C)  # expand indices
    cols = np.tile(range(C), C)

    return sp.csr_matrix((data, (rows, cols)), 
                         shape=(len(nodes_map), C), 
                         dtype=np.float32)

def separate(year):
    regex = "(\d\d)(\d)(\d)"
    return match(regex, year)

def validate(value):
    return match("{}{}".format(_REGEX_YEAR_FRAG,
                               _REGEX_TIMEZONE_FRAG),
                 value)

def normalize():
    pass
