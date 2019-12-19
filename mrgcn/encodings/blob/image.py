#!/usr/bin/python3

import logging
import base64
from io import BytesIO
from PIL import Image
from re import fullmatch

import numpy as np
from rdflib.term import Literal
from rdflib.namespace import XSD


_REGEX_BASE64 = "^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$"
_IMG_SIZE = (64, 64)
_IMG_MODE = "RGB"

logger = logging.getLogger(__name__)

def generate_features(nodes_map, node_predicate_map, config):
    """ Generate encodings for images as XSD B64-encoded literals

    Returns an N-D array A and an vector b, such that A[i] holds the matrix
    representation of the image belonging to node b[i].

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param config: configuration dictionary
    :returns: numpy array M x Ch x W x H;
                    M :- number of nodes with an image, such that M <= N
                    Ch :- number of channels for this colour space
                    W :- width number of pixels
                    H :- height in number of pixels
              numpy array 1 x M;
                    M :- number of nodes with an image, such that M <= N
    """
    logger.debug("Generating B64-encoded image encodings")
    W, H = _IMG_SIZE

    C = 128
    m = 0
    n = len(nodes_map)
    c = len([c for c in _IMG_MODE if c.isupper() or c == '1'])
    encodings = np.zeros(shape=(n, c, W, H), dtype=np.float32)
    node_idx = np.zeros(shape=(n), dtype=np.int32)

    values_max = [None for _ in range(c)]
    values_min = [None for _ in range(c)]

    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype.neq(XSD.b64string):
            # assume that all B64-encoded literals are images
            continue

        value = str(node)  ## empty value bug workaround
        if validate(value) is None:  # if invalid syntax
            continue

        blob = b64_to_img(value)
        blob = downsample(blob)

        # add to matrix structures
        a = np.array(blob, dtype=np.float32)
        if _IMG_MODE == "RGB":
            # from WxHxC to CxWxH
            a = a.transpose((0, 2, 1)).transpose((1, 0, 2))

        for ch in range(c):
            if values_max[ch] is None or a[ch].max() > values_max[ch]:
                values_max[ch] = a[ch].max()
            if values_min[ch] is None or a[ch].min() < values_min[ch]:
                values_min[ch] = a[ch].min()

        encodings[m] = a
        node_idx[m] = i
        m += 1

    logger.debug("Generated {} unique B64-encoded image encodings".format(m))

    if m <= 0:
        return None

    # normalization over channels
    for img in encodings[:m]:
        for ch in range(c):
            img[ch] = (2*(img[ch]-values_min[ch]) /
                       (values_max[ch] - values_min[ch])) - 1.0

    return [[encodings[:m], node_idx[:m], C, None, 1]]

def b64_to_img(b64string):
    im = Image.open(BytesIO(base64.decodebytes(b64string.encode())))
    if im.mode != _IMG_MODE:
        im = im.convert(_IMG_MODE)

    return im

def downsample(im):
    if im.size != _IMG_SIZE:
        return im.resize(_IMG_SIZE)

    return im

def validate(value):
    return fullmatch(_REGEX_BASE64, value)
