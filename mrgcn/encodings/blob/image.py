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


    """
    logger.debug("Generating B64-encoded image encodings")

    if True:  #config['share_weights']:
        return generate_relationwise_features(nodes_map, node_predicate_map,
                                              config)
    else:
        return generate_nodewise_features(nodes_map, config)

def generate_nodewise_features(nodes_map, config):
    """ Stack all vectors without regard of their relation
    """
    W, H = _IMG_SIZE
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
        #if validate(value) is None:  # if invalid syntax
        #    continue

        try:
            blob = b64_to_img(value)
        except:
            continue
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
    for i in range(encodings[:m].shape[0]):
        img = encodings[:m][i]
        for ch in range(c):
            img[ch] = (2*(img[ch]-values_min[ch]) /
                       (values_max[ch] - values_min[ch])) - 1.0
            encodings[:m][i] = img

    return [[encodings[:m], node_idx[:m], [-1 for _ in range(m)]]]

def generate_relationwise_features(nodes_map, node_predicate_map, config):
    W, H = _IMG_SIZE
    c = len([c for c in _IMG_MODE if c.isupper() or c == '1'])

    n = len(nodes_map)
    m = dict()
    relationwise_encodings = dict()
    node_idx = dict()
    values_min = dict()
    values_max = dict()

    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype.neq(XSD.b64string):
            # assume that all B64-encoded literals are images
            continue

        value = str(node)  ## empty value bug workaround
        #if validate(value) is None:  # if invalid syntax
        #    continue

        try:
            blob = b64_to_img(value)
        except:
            continue
        blob = downsample(blob)

        # add to matrix structures
        a = np.array(blob, dtype=np.float32)
        if _IMG_MODE == "RGB":
            # from WxHxC to CxWxH
            a = a.transpose((0, 2, 1)).transpose((1, 0, 2))

        for pred in node_predicate_map[node]:
            if pred not in relationwise_encodings.keys():
                relationwise_encodings[pred] = np.zeros(shape=(n, c, W, H),
                                                        dtype=np.float32)
                node_idx[pred] = np.zeros(shape=(n), dtype=np.int32)
                m[pred] = 0
                values_min[pred] = [None for _ in range(c)]
                values_max[pred] = [None for _ in range(c)]

            for ch in range(c):
                if values_max[pred][ch] is None or a[ch].max() > values_max[pred][ch]:
                    values_max[pred][ch] = a[ch].max()
                if values_min[pred][ch] is None or a[ch].min() < values_min[pred][ch]:
                    values_min[pred][ch] = a[ch].min()

            # add to matrix structures
            relationwise_encodings[pred][m[pred]] = a
            node_idx[pred][m[pred]] = i
            m[pred] += 1

    logger.debug("Generated {} unique B64-encoded image encodings".format(sum(m.values())))

    if len(m) <= 0:
        return None

    # normalization over channels
    for pred in relationwise_encodings.keys():
        for i in range(relationwise_encodings[pred][:m[pred]].shape[0]):
            img = relationwise_encodings[pred][:m[pred]][i]
            for ch in range(c):
                img[ch] = (2*(img[ch]-values_min[pred][ch]) /
                           (values_max[pred][ch] - values_min[pred][ch])) - 1.0
                relationwise_encodings[pred][:m[pred]][i] = img

    return [[encodings[:m[pred]], node_idx[pred][:m[pred]], [-1 for _ in range(m[pred])]]
            for pred, encodings in relationwise_encodings.items()]

def b64_to_img(b64string):
    im = Image.open(BytesIO(base64.urlsafe_b64decode(b64string.encode())))
    if im.mode != _IMG_MODE:
        im = im.convert(_IMG_MODE)

    return im

def downsample(im):
    if im.size != _IMG_SIZE:
        return im.resize(_IMG_SIZE)

    return im

def validate(value):
    return fullmatch(_REGEX_BASE64, value)
