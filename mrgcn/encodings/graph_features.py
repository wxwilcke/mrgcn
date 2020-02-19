#!/usr/bin/python3

from importlib import import_module
import logging

import numpy as np

from mrgcn.encodings.xsd.xsd_hierarchy import XSDHierarchy


logger = logging.getLogger(__name__)

ENCODINGS_PKG = "mrgcn.encodings"
EMBEDDING_FEATURES = {"xsd.boolean", "xsd.date", "xsd.gYear", "xsd.numeric"}
PREEMBEDDING_FEATURES = {"xsd.string", "blob.image", "ogc.wktLiteral"}
AVAILABLE_FEATURES = set().union(EMBEDDING_FEATURES, PREEMBEDDING_FEATURES)

def construct_features(nodes_map, knowledge_graph, feature_configs,
                      separate_literals):
    """ Construct specified features for given nodes

    Note that normalization occurs per feature, independent of the predicate
    it is linked with. Future work should separate these.

    :param nodes_map: list of node labels (URIs) with node idx {0, N}
    :param feature_config: list of features to construct, given as dicts
    :returns: numpy array N x (F * C);
                    N :- number of nodes
                    F :- number of features
                    C :- number of columns per feature
    """
    hierarchy = XSDHierarchy()
    node_predicate_map = { o:p for _,p,o in
                          knowledge_graph.triples(separate_literals=separate_literals) }
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
            encoding_sets = merge_encodings_sets(encoding_sets)

        feature_matrix.extend([_mkdense(*feature_encoding, n) for
                               feature_encoding in encoding_sets])
        features_processed.add(feature)

    X = np.empty((n,0), dtype=np.float32) if len(feature_matrix) <= 0 else np.hstack(feature_matrix)

    return [X, features]

def _mkdense(encodings, node_idx, c, encodings_length_map, _, n):
    F = np.zeros(shape=(n, c), dtype=np.float32)
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

def merge_encodings_sets(encoding_sets):
    encodings, node_idx, C, seq_length_map, nsets = encoding_sets[0]
    n = encodings.shape[0]
    c = int(C/nsets)

    encodings_merged = np.zeros(shape=(n, c), dtype=np.float32)
    for i in range(nsets):
        # assume that non-filled values are zero
        encodings_merged += encodings[:,i*c:(i+1)*c]

    return [[encodings_merged, node_idx, c, seq_length_map, 1]]
