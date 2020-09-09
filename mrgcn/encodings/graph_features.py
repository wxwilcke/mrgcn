#!/usr/bin/python3

from importlib import import_module
import logging

import numpy as np
import scipy.sparse as sp

from mrgcn.encodings.xsd.xsd_hierarchy import XSDHierarchy
from mrgcn.tasks.utils import (mkbatches,
                               mkbatches_varlength,
                               remove_outliers,
                               trim_outliers)

logger = logging.getLogger(__name__)

ENCODINGS_PKG = "mrgcn.encodings"
EMBEDDING_FEATURES = {"xsd.boolean", "xsd.date", "xsd.dateTime", "xsd.gYear", "xsd.numeric"}
PREEMBEDDING_FEATURES = {"xsd.string", "xsd.anyURI", "blob.image", "ogc.wktLiteral"}
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
    X_width = 0
    modules_config = list()
    preembeddings = list()
    for datatype in set.intersection(set(features_enabled),
                                     set(features.keys()),
                                     PREEMBEDDING_FEATURES):
        feature_config = next((conf for conf in feature_configs
                               if conf['datatype'] == datatype),
                              None)
        weight_sharing = feature_config['share_weights']
        embedding_dim = feature_config['embedding_dim']

        encoding_sets = features.pop(datatype, list())
        if weight_sharing:
            logger.debug("weight sharing enabled for {}".format(datatype))
            if datatype in ["xsd.string", "xsd.anyURI", "ogc.wktLiteral"]:
                encoding_sets = merge_sparse_encodings_sets(encoding_sets)
            elif datatype in ["blob.image"]:
                encoding_sets = merge_img_encoding_sets(encoding_sets)
            else:
                pass

        nsets = len(encoding_sets)
        for encodings, node_idx, seq_lengths in encoding_sets:
            if datatype in ["xsd.string", "xsd.anyURI"]:
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

                modules_config.append((datatype, (feature_config['num_batches'],
                                                  feature_size,
                                                  embedding_dim,
                                                  model_size)))
            if datatype in ["ogc.wktLiteral"]:
                # stored as list of arrays
                feature_dim = 0  # set to 1 for RNN
                feature_size = encodings[0].shape[feature_dim]
                modules_config.append((datatype, (feature_config['num_batches'],
                                                  feature_size,
                                                  embedding_dim)))
            if datatype in ["blob.image"]:
                # stored as tensor
                modules_config.append((datatype, (feature_config['num_batches'],
                                                  encodings.shape[1:],
                                                  embedding_dim)))

            X_width += embedding_dim

        # deal with outliers?
        if datatype in ["ogc.wktLiteral", "xsd.string", "xsd.anyURI"]:
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
                                                  num_batches=feature_config['num_batches'])))
            elif datatype in ["ogc.wktLiteral", "xsd.string", "xsd.anyURI"]:
                encoding_sets_batched.append((f, mkbatches_varlength(*f,
                                                            num_batches=feature_config['num_batches'])))

        preembeddings.append((datatype, encoding_sets_batched))

    return (preembeddings, modules_config, X_width)

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

def _mkdense(encodings, node_idx, encodings_length_map, n):
    """ Return N x M matrix with N := NUM_NODES and M := NUM_COLS
        Use node index to map encodings to correct nodes
    """
    F = np.zeros(shape=(n, encodings.shape[1]), dtype=np.float32)
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

def merge_sparse_encodings_sets(encoding_sets):
    node_idc = np.concatenate([node_idx for _, node_idx, _ in encoding_sets])
    node_idc_unique, node_idc_counts = np.unique(node_idc, return_counts=True)
    node_idc_mult = node_idc_unique[node_idc_counts > 1]

    N = node_idc_unique.shape[0]  # N <= NUM_NODES

    encodings_merged = [None for _ in range(N)]
    node_idx_merged = node_idc_unique
    seq_length_merged = np.zeros(shape=N, dtype=np.int32)

    merged_idx_map = {v:i for i,v in enumerate(node_idx_merged)}
    merged_idx_mult = {merged_idx_map[v]:0 for v in node_idc_mult}
    for encodings, node_index, seq_length in encoding_sets:
        # assume that non-filled values are zero
        for i in range(node_index.shape[0]):
            idx = node_index[i]
            merged_idx = merged_idx_map[idx]

            enc = encodings[i]
            enc_length = seq_length[i]
            if idx in node_idc_mult:
                if merged_idx_mult[merged_idx] > 0:
                    # average vectors for nodes that occur more than once
                    enc_length = max(enc_length,
                                     seq_length_merged[merged_idx])

                    enc_visited = encodings_merged[merged_idx]
                    shape = (max(enc.shape[0], enc_visited.shape[0]),
                             max(enc.shape[1], enc_visited.shape[1]))
                    enc = sp.coo_matrix((np.concatenate([enc.data, enc_visited.data]),
                                        (np.concatenate([enc.row, enc_visited.row]),
                                         np.concatenate([enc.col, enc_visited.col]))),
                                        shape=shape)

                merged_idx_mult[merged_idx] += 1

            encodings_merged[merged_idx] = enc
            seq_length_merged[merged_idx] = enc_length

    for idx, n in merged_idx_mult.items():
        # average vectors for nodes that occur more than once
        enc = encodings_merged[idx]
        data = enc.data / n
        encodings_merged[idx] = sp.coo_matrix((data, (enc.row, enc.col)),
                                              shape=enc.shape)

    return [[encodings_merged, node_idx_merged, seq_length_merged]]

def merge_encoding_sets(encoding_sets):
    """ Merge encoding sets into a single set. Entries for the same node are
    merged by averaging the values, which actually only matters if node
    encodings depend on more than their content (eg content+predicates),
    which they do not atm.
    """
    if len(encoding_sets) <= 1:
        return encoding_sets

    node_idc = np.concatenate([node_idx for _, node_idx, _ in encoding_sets])
    node_idc_unique, node_idc_counts = np.unique(node_idc, return_counts=True)
    node_idc_mult = node_idc_unique[node_idc_counts > 1]

    N = node_idc_unique.shape[0]  # N <= NUM_NODES
    M = max([enc.shape[1] for enc, _, _ in encoding_sets])  # M = MAX(NUM_COLS per set)

    encodings_merged = np.zeros(shape=(N, M), dtype=np.float32)
    node_idx_merged = node_idc_unique
    seq_length_merged = np.zeros(shape=N, dtype=np.int32)

    merged_idx_map = {v:i for i,v in enumerate(node_idx_merged)}
    merged_idx_mult = {merged_idx_map[v]:0 for v in node_idc_mult}
    for encodings, node_index, seq_length in encoding_sets:
        # assume that non-filled values are zero
        for i in range(node_index.shape[0]):
            idx = node_index[i]
            merged_idx = merged_idx_map[idx]

            enc = encodings[i]
            if idx in node_idc_mult:
                # average vectors for nodes that occur more than once
                enc += encodings_merged[merged_idx,:encodings.shape[1]]
                merged_idx_mult[merged_idx] += 1

            encodings_merged[merged_idx,:encodings.shape[1]] = enc
            if seq_length is not None:
                # use max length for nodes that occur more than once
                seq_length_merged[merged_idx] = max(seq_length_merged[merged_idx],
                                                    seq_length[i])

    for idx, n in merged_idx_mult.items():
        # average vectors for nodes that occur more than once
        encodings_merged[idx] = encodings_merged[idx] / n

    return [[encodings_merged, node_idx_merged, seq_length_merged]]

def merge_img_encoding_sets(encoding_sets):
    """ Merge encoding sets into a single set. Entries for the same node are
    merged by averaging the values, which actually only matters if node
    encodings depend on more than their content (eg content+predicates),
    which they do not atm.
    """
    if len(encoding_sets) <= 1:
        return encoding_sets

    node_idc = np.concatenate([node_idx for _, node_idx, _ in encoding_sets])
    node_idc_unique, node_idc_counts = np.unique(node_idc, return_counts=True)
    node_idc_mult = node_idc_unique[node_idc_counts > 1]

    N = node_idc_unique.shape[0]  # N <= NUM_NODES
    c, W, H = encoding_sets[0][0].shape[1:]  # assume same image size on all

    encodings_merged = np.zeros(shape=(N, c, W, H), dtype=np.float32)
    node_idx_merged = node_idc_unique

    merged_idx_map = {v:i for i,v in enumerate(node_idx_merged)}
    merged_idx_mult = {merged_idx_map[v]:0 for v in node_idc_mult}
    for encodings, node_index, _ in encoding_sets:
        # assume that non-filled values are zero
        for i in range(node_index.shape[0]):
            idx = node_index[i]
            merged_idx = merged_idx_map[idx]

            enc = encodings[i]
            if idx in node_idc_mult:
                # average vectors for nodes that occur more than once
                enc += encodings_merged[merged_idx]
                merged_idx_mult[merged_idx] += 1

            encodings_merged[merged_idx] = enc

    for idx, n in merged_idx_mult.items():
        # average vectors for nodes that occur more than once
        encodings_merged[idx] = encodings_merged[idx] / n

    return [[encodings_merged, node_idx_merged, None]]

def stack_encoding_sets(encoding_sets):
    """ Stack encoding sets horizontally. Entries for the same node are
    placed on the same row.
    """
    if len(encoding_sets) <= 1:
        return encoding_sets

    node_idc = np.concatenate([node_idx for _, node_idx, _ in encoding_sets])
    node_idc_unique = np.unique(node_idc)

    N = node_idc_unique.shape[0]  # N <= NUM_NODES
    M = sum([enc.shape[1] for enc, _, _ in encoding_sets])  # M = SUM(NUM_COLS per set)

    encodings_merged = np.zeros(shape=(N, M), dtype=np.float32)
    node_idx_merged = node_idc_unique
    seq_length_merged = np.repeat([M], repeats=N)

    merged_idx_map = {v:i for i,v in enumerate(node_idx_merged)}
    j = 0
    for encodings, node_index, seq_length in encoding_sets:
        m = encodings.shape[1]
        # assume that non-filled values are zero
        for k in range(node_index.shape[0]):
            idx = node_index[k]
            merged_idx = merged_idx_map[idx]

            enc = encodings[k]
            encodings_merged[merged_idx,j:j+m] = enc

        j += m

    return [[encodings_merged, node_idx_merged, seq_length_merged]]
