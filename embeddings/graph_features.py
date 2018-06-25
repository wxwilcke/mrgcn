#!/usr/bin/python3

from importlib import import_module
import logging

import scipy.sparse as sp


logger = logging.getLogger(__name__)

EMBEDDINGS_PKG = "embeddings.xsd"
AVAILABLE_FEATURES = ["gYear"]

def construct_features(nodes_map, feature_list):
    """ Construct specified features for given nodes 

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param feature_list: list of features to construct, given as module names (str)
    :returns: scipy sparse matrix N x (F * C); 
                    N :- number of nodes
                    F :- number of features
                    C :- number of columns per feature 
    """
    features = sp.csr_matrix((len(nodes_map), 0))
    for feature_name in feature_list:
        if feature_name not in AVAILABLE_FEATURES:
            logger.debug("Specified feature not available: {}".format(feature_name))
            continue

        # dynamically load module
        module = import_module("{}.{}".format(EMBEDDINGS_PKG, feature_name))
        feature = module.generate_features(nodes_map)

        logger.debug("Concatenating {} features to X".format(feature_name))
        # stack new features to existing ones
        features = sp.hstack([features, feature], format='csr')

    return features
