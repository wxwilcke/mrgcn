#!/usr/bin/python3

from importlib import import_module
import logging

import numpy as np

from mrgcn.encodings.xsd.xsd_hierarchy import XSDHierarchy
from mrgcn.tasks.utils import (mkbatches,
                               mkbatches_varlength,
                               remove_outliers,
                               trim_outliers)

logger = logging.getLogger(__name__)

ENCODINGS_PKG = "mrgcn.encodings"
EMBEDDING_FEATURES = {"xsd.boolean", "xsd.date", "xsd.gYear", "xsd.numeric"}
PREEMBEDDING_FEATURES = {"xsd.string", "blob.image", "ogc.wktLiteral"}
AVAILABLE_FEATURES = set.union(EMBEDDING_FEATURES, PREEMBEDDING_FEATURES)

def construct_features(nodes_map, knowledge_graph, feature_configs,
                      separate_literals):
    """ Construct specified features for given nodes

    Note that normalization occurs per feature, independent of the predicate
    it is linked with. Future work should separate these.

    """
    hierarchy = XSDHierarchy()
    node_predicate_map = dict()
    for _,p,o in knowledge_graph.triples(separate_literals=separate_literals):
        if o not in node_predicate_map.keys():
            node_predicate_map[o] = set()
        node_predicate_map[o].add(p)

    features = dict()
    for feature_config in feature_configs:
        if not feature_config['include']:
            continue

        feature_name = feature_config['datatype']
        feature = feature_module(hierarchy, feature_name)
        if feature is None:
            logger.debug("Specified feature not available: {}".format(feature_name))
            continue

        # dynamically load module
        module = import_module("{}.{}".format(ENCODINGS_PKG, feature))
        feature_encoding = module.generate_features(nodes_map,
                                                    node_predicate_map,
                                                    feature_config)

        if feature_encoding is not None:
            features[feature_name] = feature_encoding

    return features

def feature_module(hierarchy, feature_name):
    for feature in AVAILABLE_FEATURES:
        # prefer more tailored module
        if feature == feature_name:
            return feature

    if not feature_name.startswith("xsd"):
        return None

    feature_name = feature_name[4:]
    for feature in AVAILABLE_FEATURES:
        if not feature.startswith("xsd"):
            continue
        if hierarchy.subtypeof(feature[4:], feature_name):
            return feature

    return None

def construct_preembeddings(features, features_enabled, n, nepoch, feature_configs):
    C = 0
    modules_config = list()
    preembeddings = list()
    for datatype in set.intersection(set(features_enabled),
                                     set(features.keys()),
                                     PREEMBEDDING_FEATURES):
        feature_config = next((conf for conf in feature_configs
                               if conf['datatype'] == datatype),
                              None)
        weight_sharing = feature_config['share_weights']

        encoding_sets = features.pop(datatype, list())
        if weight_sharing and datatype == "xsd.string":
            # note: images and geometries always share weights atm
            logger.debug("weight sharing enabled for {}".format(datatype))
            encoding_sets = merge_sparse_encodings_sets(encoding_sets)

        for encodings, node_idx, c, seq_lengths, nsets in encoding_sets:
            if datatype in ["xsd.string"]:
                # stored as list of arrays
                feature_dim = 0
                feature_size = encodings[0].shape[feature_dim]

                model_size = "M"  # medium, seq length >= 12
                if not weight_sharing or nsets <= 1:
                    seq_length_min = min(seq_lengths)
                    if seq_length_min < 20:
                        model_size = "S"
                    elif seq_length_min < 50:
                        model_size = "M"
                    else:
                        model_size = "L"

                modules_config.append((datatype, (feature_config['passes_per_batch'],
                                                  feature_size,
                                                  c,
                                                  model_size)))
            if datatype in ["ogc.wktLiteral"]:
                # stored as list of arrays
                feature_dim = 0  # set to 1 for RNN
                feature_size = encodings[0].shape[feature_dim]
                modules_config.append((datatype, (feature_config['passes_per_batch'],
                                                  feature_size,
                                                  c)))
            if datatype in ["blob.image"]:
                # stored as tensor
                modules_config.append((datatype, (feature_config['passes_per_batch'],
                                                  encodings.shape[1:],
                                                  c)))

            C += c

        # deal with outliers?
        if datatype in ["ogc.wktLiteral", "xsd.string"]:
            if feature_config['remove_outliers']:
                encoding_sets = [remove_outliers(*f) for f in encoding_sets]
            if feature_config['trim_outliers']:
                feature_dim = 0  # set to 1 for RNN
                encoding_sets = [trim_outliers(*f, feature_dim) for f in encoding_sets]

        nepoch
        encoding_sets_batched = list()
        for f in encoding_sets:
            if datatype == "blob.image":
                encoding_sets_batched.append((f, mkbatches(*f,
                                                  nepoch=nepoch,
                                                  passes_per_batch=feature_config['passes_per_batch'])))
            elif datatype == "ogc.wktLiteral":
                encoding_sets_batched.append((f, mkbatches_varlength(*f,
                                                            nepoch=nepoch,
                                                            passes_per_batch=feature_config['passes_per_batch'])))
            elif datatype == "xsd.string":
                encoding_sets_batched.append((f, mkbatches_varlength(*f,
                                                            nepoch=nepoch,
                                                            passes_per_batch=feature_config['passes_per_batch'])))

        preembeddings.append((datatype, encoding_sets_batched))

    return (preembeddings, modules_config, C)

def construct_feature_matrix(features, features_enabled, n, feature_configs):
    feature_matrix = list()
    features_processed = set()
    for feature in features_enabled:
        if feature not in features.keys():
            logging.debug("=> WARNING: feature {} not in dataset".format(feature))
            continue

        if feature in PREEMBEDDING_FEATURES:
            # these require additional processing before they can be
            # concatenated to X
            continue

        feature_config = next((conf for conf in feature_configs
                               if conf['datatype'] == feature),
                              None)
        encoding_sets = features[feature]
        if feature_config['share_weights']:
            logger.debug("weight sharing enabled for {}".format(feature))
            encoding_sets = merge_encoding_sets(encoding_sets)
        else:
            encoding_sets = stack_encoding_sets(encoding_sets)

        feature_matrix.extend([_mkdense(*feature_encoding, n) for
                               feature_encoding in encoding_sets])
        features_processed.add(feature)

    X = np.empty((n,0), dtype=np.float32) if len(feature_matrix) <= 0 else np.hstack(feature_matrix)

    return X

def _mkdense(encodings, node_idx, m, encodings_length_map, _, n):
    """ Return N x M matrix with N := NUM_NODES and M := NUM_COLS
        Use node index to map encodings to correct nodes
    """
    F = np.zeros(shape=(n, m), dtype=np.float32)
    F[node_idx] = encodings

    return F

def features_included(config):
    features = set()

    if 'features' not in config['graph']:
        return features
    feature_configs = config['graph']['features']

    for feature_config in feature_configs:
        if not feature_config['include']:
            continue

        features.add(feature_config['datatype'])

    return features

def merge_sparse_encodings_sets(encodings):
    encodings_merged = list()
    node_idx_merged = list()
    seq_lengths_merged = list()
    C_max = 0

    for encoding_set, node_idx, C, seq_length_map, _, in encodings:
        encodings_merged.extend(encoding_set)
        node_idx_merged.extend(node_idx)
        seq_lengths_merged.extend(seq_length_map)

        if C > C_max:
            C_max = C

    return [[encodings_merged, node_idx_merged, C_max, seq_lengths_merged, 1]]

def merge_encoding_sets(encoding_sets):
    """ Merge encodings sets

        returns N x M encodings with N <= NUM_NODES and M = MAX(N_COLS per set)
    """
    shapes = [encodings.shape for encodings, _, _, _, _ in encoding_sets]
    N = sum([shape[0] for shape in shapes])
    M = max([shape[1] for shape in shapes])

    encodings_merged = np.zeros(shape=(N, M), dtype=np.float32)
    node_idx_merged = np.zeros(shape=(N), dtype=np.int32)
    seq_length_merged = list()

    i = 0
    for encodings, node_index, _, seq_length, _ in encoding_sets:
        # assume that non-filled values are zero
        encodings_merged[i:i+encodings.shape[0],:encodings.shape[1]] = encodings
        node_idx_merged[i:i+encodings.shape[0]] = node_index
        if seq_length is not None:
            seq_length_merged.extend(seq_length)

        i += encodings.shape[0]

    return [[encodings_merged, node_idx_merged, M, seq_length_merged, 1]]

def stack_encoding_sets(encoding_sets):
    """ Stack encodings sets horizontally

        returns N x M encodings with N <= NUM_NODES and M = SUM(N_COLS per set)
    """
    shapes = [encodings.shape for encodings, _, _, _, _ in encoding_sets]
    N = sum([shape[0] for shape in shapes])
    M = sum([shape[1] for shape in shapes])

    encodings_merged = np.zeros(shape=(N, M), dtype=np.float32)
    node_idx_merged = np.zeros(shape=(N), dtype=np.int32)
    seq_length_merged = list()

    i = 0
    j = 0
    for encodings, node_index, _, seq_length, _ in encoding_sets:
        # assume that non-filled values are zero
        encodings_merged[i:i+encodings.shape[0], j:j+encodings.shape[1]] = encodings
        node_idx_merged[i:i+encodings.shape[0]] = node_index
        if seq_length is not None:
            seq_length_merged.extend(seq_length)

        i += encodings.shape[0]
        j += encodings.shape[1]

    return [[encodings_merged, node_idx_merged, M, seq_length_merged, 1]]
