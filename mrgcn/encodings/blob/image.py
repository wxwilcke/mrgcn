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

def generate_features(nodes_map, config):
    """ Generate features for images as XSD B64-encoded literals

    Returns an N-D array A and an vector b, such that A[i] holds the matrix
    representation of the image belonging to node b[i].

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param config: configuration dictionary
    :returns: numpy array M x C x W x H;
                    M :- number of nodes with an image, such that M <= N
                    C :- number of channels for this colour space
                    W :- width number of pixels
                    H :- height in number of pixels
              numpy array 1 x M;
                    M :- number of nodes with an image, such that M <= N
    """
    logger.debug("Generating B64-encoded image features")
    W, H = _IMG_SIZE

    m = 0
    n = len(nodes_map)
    C = len([c for c in _IMG_MODE if c.isupper() or c == '1'])
    images = np.zeros(shape=(n, C, W, H), dtype=np.float32)
    node_idx = np.zeros(shape=(n), dtype=np.int32)

    channel_means = np.zeros(C, dtype=np.float32)
    channel_stdev = np.zeros(C, dtype=np.float32)

    for node, i in nodes_map.items():
        if type(node) is not Literal:
            continue
        if node.datatype is None or node.datatype.neq(XSD.b64string):
            # assume that all B64-encoded literals are images
            continue

        node._value = node.__str__()  ## empty value bug workaround
        value = validate(node.value)
        if value is None:  # if invalid syntax
            continue

        blob = b64_to_img(value)
        blob = downsample(blob)

        # add to matrix structures
        a = np.array(blob, dtype=np.float32)
        if _IMG_MODE == "RGB":
            # from WxHxC to CxWxH
            a = a.transpose((0, 2, 1)).transpose((1, 0, 2))

        images[m] = a
        node_idx[m] = i
        m += 1

        for c in range(C):
            # store means and stdevs for normalization
            channel_means[c] += np.mean(a[c])
            channel_stdev[c] += np.std(a[c])

    logger.debug("Generated {} unique B64-encoded image features".format(m))

    #inplace normalization over channels
    if config['normalize']:
        normalize(images[:m], C, channel_means/m, channel_stdev/m)

    return [images[:m], node_idx[:m]]

def normalize(images, C, means, stdevs):
    for img in images:
        for c in range(C):
            img[c] = (img[c] - means[c]) / stdevs[c]

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
