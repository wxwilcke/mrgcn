#!/usr/bin/python3

import logging
import base64
from io import BytesIO
from PIL import Image

import numpy as np
from rdflib.term import Literal, URIRef
from rdflib import Namespace


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
    n = len(nodes_map)
    m = dict()
    encodings = dict()
    node_idx = dict()

    im_mode = config['transform']['mode']
    c = len(im_mode)  # assume each letter equals one channel

    im_size_cropped = config['transform']['centerCrop']
    im_size_base = config['transform']['resizeSize']
    im_size_interpolation_mode = getattr(Image, config['transform']['interpolationMode'])
    
    # multiply by 255 to translate floats in [0,1] to pixels in [0,255]
    mean_values = np.array(config['transform']['mean']) * 255
    std_values = np.array(config['transform']['std']) * 255

    failed = 0
    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype.neq(_KGB_NAMESPACE.base64Image):
            continue

        im = None
        try:
            value = str(node)
            im = b64_to_img(value)
            if im.mode != im_mode:
                im = im.convert(im_mode)
            im = resize(im, im_size_base, im_size_interpolation_mode)
            im = centerCrop(im, im_size_cropped)
        except ValueError:
            failed += 1
            continue

        # add to matrix structures
        a = np.array(im, dtype=np.float32)  # HxWxC

        if im_mode == "RGB":
            a = a.transpose((2, 0, 1))  # change to CxHxW

        # normalize values along channels
        a = (a - mean_values[:, None, None]) / std_values[:, None, None]

        for p in node_predicate_map[node]:
            if p not in encodings.keys():
                encodings[p] = np.empty(shape=(n, c, im_size_cropped,
                                                     im_size_cropped),
                                        dtype=np.float32)
                node_idx[p] = np.empty(shape=(n), dtype=np.int32)
                m[p] = 0

            # add to matrix structures
            idx = m[p]
            encodings[p][idx] = a
            node_idx[p][idx] = i
            m[p] = idx + 1

    msum = sum(m.values())
    logger.debug("Generated {} unique B64-encoded image encodings ({} failed)".format(msum,
                                                                                      failed))

    if msum <= 0:
        return None

    return [[encodings[p][:m[p]], node_idx[p][:m[p]], -np.ones(m[p])]
            for p in encodings.keys()]

def b64_to_img(b64string):
    im = Image.open(BytesIO(base64.urlsafe_b64decode(b64string.encode())))

    return im

def resize(im, size, interpolate_mode):
    w, h = im.size
    if w == size and h == size:
        return im
    elif w == h:
        return im.resize((size, size), interpolate_mode)
    elif w > h:
        return im.resize(((size * w)//h, size), interpolate_mode)
    else:  # h < w
        return im.resize((size, (h * size)//w), interpolate_mode)

def centerCrop(im, size):
    w, h = im.size

    left = int(w/2 - size/2)
    top = int(h/2 - size/2)
    right = left + size
    bottom = top + size

    return im.crop((left, top, right, bottom))
