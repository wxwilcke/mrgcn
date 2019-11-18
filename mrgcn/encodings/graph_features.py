#!/usr/bin/python3

from importlib import import_module
import logging

import numpy as np

from mrgcn.encodings.xsd.xsd_hierarchy import XSDHierarchy


logger = logging.getLogger(__name__)

ENCODINGS_PKG = "mrgcn.encodings"
EMBEDDING_FEATURES = {"xsd.gYear", "xsd.numeric"}
PREEMBEDDING_FEATURES = {"xsd.string", "blob.image"}
AVAILABLE_FEATURES = set().union(EMBEDDING_FEATURES, PREEMBEDDING_FEATURES)

def construct_features(nodes_map, knowledge_graph, feature_configs):
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
    node_predicate_map = { o:p for _,p,o in knowledge_graph.triples() }
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
        module = import_module("{}.{}".format(ENCODINGS_PKG, feature_name))
        encodings, node_idx, C, encoding_length_map = module.generate_features(nodes_map,
                                                                      node_predicate_map,
                                                                      feature_config)

        features[feature_name] = [encodings, node_idx, C, encoding_length_map]

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

def construct_feature_matrix(features, n):
    feature_matrix = list()
    features_processed = set()
    for feature in features.keys():
        if feature in PREEMBEDDING_FEATURES:
            # these require additional processing before they can be
            # concatenated to X
            continue

        feature_matrix.append(_mkdense(*features[feature], n))
        features_processed.add(feature)

    for feature in features_processed:
        # save some memory
        del features[feature]

    return [np.hstack(feature_matrix), features]

def _mkdense(encodings, node_idx, c, encodings_length_map, n):
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
