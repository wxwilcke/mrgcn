#!/usr/bin/python3

import logging
import base64
from io import BytesIO
from PIL import Image
from re import fullmatch

import numpy as np
from rdflib.term import Literal, URIRef
from rdflib import Namespace


_REGEX_BASE64 = "^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$"
_IMG_SIZE = 256
_IMG_CROP = 224
_IMG_MODE = "RGB"

_KGB_NAMESPACE = Namespace(URIRef("http://kgbench.info/dt#"))

logger = logging.getLogger(__name__)


def generate_features(nodes_map, node_predicate_map, config):
    """ Generate encodings for images as XSD B64-encoded literals

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param node_predicate_map: dictionary of node labels (URIs): {predicates}
    :param config: configuration dictionary
    :returns: list of length P with lists Q of length 3;
                P :- number of predicates that link to nodes with this feature
                Q :- [encodings, node_idx, enc_lengths];
                    enc :- (M, CH, H, W) tensor holding images;
                        M :- number of images for this predicate, such that M <= N
                        CH :- number of channels
                        H :- height
                        W :- width
                    node_idx :- numpy vector of length M, mapping enc index to node id
                    enc_lengths :- numpy array of -1's of length M

    """
    logger.debug("Generating B64-encoded image encodings")

    return generate_relationwise_features(nodes_map, node_predicate_map,
                                          config)


def generate_relationwise_features(nodes_map, node_predicate_map, config):
    c = len([c for c in _IMG_MODE if c.isupper() or c == '1'])

    n = len(nodes_map)
    m = dict()
    encodings = dict()
    node_idx = dict()
    values_min = dict()
    values_max = dict()

    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype.neq(_KGB_NAMESPACE.base64Image):
            continue

        im = None
        try:
            value = str(node)
            im = b64_to_img(value)
            im = resize(im)
            im = centerCrop(im)
        except ValueError:
            continue

        # add to matrix structures
        a = np.array(im, dtype=np.float32)
        a /= 255  # all values between 0 and 1
        if _IMG_MODE == "RGB":
            # from WxHxC to CxHxW
            a = a.transpose((2, 0, 1))

        for p in node_predicate_map[node]:
            if p not in encodings.keys():
                encodings[p] = np.empty(size=(n, c, _IMG_CROP, _IMG_CROP))
                node_idx[p] = np.empty(shape=(n), dtype=np.int32)
                m[p] = 0
                values_min[p] = [None for _ in range(c)]
                values_max[p] = [None for _ in range(c)]

            for ch in range(c):
                if values_max[p][ch] is None or a[ch].max() > values_max[p][ch]:
                    values_max[p][ch] = a[ch].max()
                if values_min[p][ch] is None or a[ch].min() < values_min[p][ch]:
                    values_min[p][ch] = a[ch].min()

            # add to matrix structures
            idx = m[p]
            encodings[p][idx] = a
            node_idx[p][idx] = i
            m[p] = idx + 1

    msum = sum(m.values())
    logger.debug("Generated {} unique B64-encoded image encodings".format(msum))

    if msum <= 0:
        return None

    # normalization over channels
    for p in encodings.keys():
        for i in range(encodings[p].shape[0]):
            img = encodings[p][i]
            for ch in range(c):
                img[ch] = (2*(img[ch]-values_min[p][ch]) /
                           (values_max[p][ch] - values_min[p][ch])) - 1.0

            encodings[p][i] = img

    return [[encodings[p][:m[p]], node_idx[p][:m[p]], -np.ones(m[p])]
            for p in encodings.keys()]


def b64_to_img(b64string):
    im = Image.open(BytesIO(base64.urlsafe_b64decode(b64string.encode())))
    if im.mode != _IMG_MODE:
        im = im.convert(_IMG_MODE)

    return im


def resize(im):
    w, h = im.size
    if w == _IMG_SIZE and h == _IMG_SIZE:
        return im
    elif w == h:
        return im.resize((_IMG_SIZE, _IMG_SIZE))
    elif w > h:
        return im.resize(((_IMG_SIZE * w)//h, _IMG_SIZE))
    else:  # h < w
        return im.resize((_IMG_SIZE, (h * _IMG_SIZE)//w))


def centerCrop(im):
    w, h = im.size

    left = int(w/2 - _IMG_CROP/2)
    top = int(h/2 - _IMG_CROP/2)
    right = left + _IMG_CROP
    bottom = top + _IMG_CROP

    return im.crop((left, top, right, bottom))


def validate(value):
    return fullmatch(_REGEX_BASE64, value)
