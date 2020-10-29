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
EMBEDDING_FEATURES = {"xsd.boolean", "xsd.numeric"}
PREEMBEDDING_FEATURES = {"xsd.string", "xsd.anyURI", "blob.image",
                         "ogc.wktLiteral", "xsd.date", "xsd.dateTime",
                         "xsd.gYear"}
AVAILABLE_FEATURES = set.union(EMBEDDING_FEATURES, PREEMBEDDING_FEATURES)

def construct_features(nodes_map, knowledge_graph, feature_configs,
                      separate_literals):
    """ Construct specified features for given nodes
    """
    hierarchy = XSDHierarchy()

    # keep track which predicates are used to link to which objects
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
        # feature_encoding is a list for every predicate with this feature as
        # range, with each member being also a list L of length 3 with L[0] the 
        # vectorized encodings (array if fixed length, list of arrays
        # otherwise), L[1] an array with node indices that map the encodings to the
        # correct graph' nodes, and L[2] an array with the lengths of the
        # encodings (uniform if fixed length).

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

def construct_preembeddings(F, features_enabled, feature_configs):
    embeddings_width = 0
    modules_config = list()
    preembeddings = list()
    optimizer_config = list()
    for datatype in set.intersection(set(features_enabled),
                                     set(F.keys()),
                                     PREEMBEDDING_FEATURES):
        feature_config = next((conf for conf in feature_configs
                               if conf['datatype'] == datatype),
                              None)
        embedding_dim = feature_config['embedding_dim']

        # parameters to pass to optimizer
        optim_params = dict()
        for param_name, param_value in feature_config.items():
            if not param_name.startswith('optim_'):
                # not an optimizer parameter
                continue

            param_name = param_name.lstrip('optim_')
            optim_params[param_name] = param_value

        optimizer_config.append((datatype, optim_params))

        encoding_sets = F.pop(datatype, list())
        weight_sharing = feature_config['share_weights']
        if weight_sharing:
            logger.debug("weight sharing enabled for {}".format(datatype))
            if datatype in ["blob.image"]:
                encoding_sets = merge_img_encoding_sets(encoding_sets)
            elif datatype in ["xsd.string", "xsd.anyURI", "ogc:wktLiteral"]:
                encoding_sets = merge_sparse_encodings_sets(encoding_sets)
            elif datatype in ["xsd.date", "xsd.dateTime", "xsd.gYear"]:
                encoding_sets = merge_encoding_sets(encoding_sets)
            else:
                logger.warning("Unsupported datatype %s" % datatype)

        num_encoding_sets = len(encoding_sets)
        for encodings, node_idx, seq_lengths in encoding_sets:
            if datatype in ["xsd.date", "xsd.dateTime", "xsd.gYear"]:
                feature_dim = 1
                feature_size = encodings.shape[feature_dim]
                modules_config.append((datatype, (feature_size,
                                                  embedding_dim)))
            if datatype in ["xsd.string", "xsd.anyURI"]:
                # stored as list of arrays (vocab x length)
                feature_dim = 0
                feature_size = encodings[0].shape[feature_dim]

                # adjust model size to 'best' fit sequence lengths
                model_size = "M"  # medium, seq length >= 12
                if not weight_sharing or num_encoding_sets <= 1:
                    seq_length_min = min(seq_lengths)
                    if seq_length_min < 20:
                        model_size = "S"
                    elif seq_length_min < 50:
                        model_size = "M"
                    else:
                        model_size = "L"

                modules_config.append((datatype, (feature_size,
                                                  embedding_dim,
                                                  model_size)))
            if datatype in ["ogc.wktLiteral"]:
                # stored as list of arrays (point_repr x num_points)
                feature_dim = 0  # set to 1 for RNN
                feature_size = encodings[0].shape[feature_dim]
                modules_config.append((datatype, (feature_size,
                                                  embedding_dim)))
            if datatype in ["blob.image"]:
                # stored as tensor (num_images x num_channels x width x height)
                modules_config.append((datatype, (encodings.shape[1:],
                                                  embedding_dim)))

            embeddings_width += embedding_dim

        # deal with outliers?;
        if 'remove_outliers' in feature_config.keys() and feature_config['remove_outliers']:
            if datatype in ["ogc.wktLiteral", "xsd.string", "xsd.anyURI"]:
                encoding_sets = [remove_outliers(*f) for f in encoding_sets]
            else:
                raise Warning("Outlier removal not supported for datatype %s" %
                              datatype)

        if 'trim_outliers' in feature_config.keys() and feature_config['trim_outliers']:
            if datatype in ["ogc.wktLiteral", "xsd.string", "xsd.anyURI"]:
                feature_dim = 0  # set to 1 for RNN
                encoding_sets = [trim_outliers(*f, feature_dim) for f in encoding_sets]
            else:
                raise Warning("Outlier trimming not supported for datatype %s" %
                              datatype)


        num_batches = 1 if 'num_batches' not in feature_config.keys()\
                        else feature_config['num_batches']
        encoding_sets_batched = list()
        for encodings, node_idc, enc_lengths in encoding_sets:
            if datatype in ["blob.image", "xsd.date", "xsd.dateTime",
                            "xsd.gYear"]:
                batches = mkbatches(encodings,
                                    node_idc,
                                    num_batches=num_batches)
            elif datatype in ["ogc.wktLiteral", "xsd.string", "xsd.anyURI"]:
                batches = mkbatches_varlength(encodings,
                                              node_idc,
                                              enc_lengths,
                                              num_batches=num_batches)
            encoding_sets_batched.append((encodings, batches))

        preembeddings.append((datatype, encoding_sets_batched))

    return (preembeddings, modules_config, optimizer_config, embeddings_width)

def construct_feature_matrix(F, features_enabled, num_nodes, feature_configs):
    feature_matrix = list()
    for feature in features_enabled:
        if feature not in F.keys():
            logging.debug("=> WARNING: feature {} not in dataset".format(feature))
            continue

        if feature in PREEMBEDDING_FEATURES:
            # these require additional processing before they can be
            # concatenated to X
            continue

        feature_config = next((conf for conf in feature_configs
                               if conf['datatype'] == feature),
                              None)
        encoding_sets = F[feature]
        if feature_config['share_weights']:
            logger.debug("weight sharing enabled for {}".format(feature))
            encoding_sets = merge_encoding_sets(encoding_sets)
        else:
            encoding_sets = stack_encoding_sets(encoding_sets)

        feature_matrix.extend([_mkdense(*feature_encoding, num_nodes) for
                               feature_encoding in encoding_sets])

    X = np.empty((num_nodes, 0), dtype=np.float32) if len(feature_matrix) <= 0\
        else np.hstack(feature_matrix)

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
    if len(encoding_sets) <= 1:
        return encoding_sets

    node_idc = np.concatenate([node_idx for _, node_idx, _ in encoding_sets])
    node_idc_unique, node_idc_counts = np.unique(node_idc, return_counts=True)
    node_idc_mult = node_idc_unique[node_idc_counts > 1]

    N = node_idc_unique.shape[0]  # N <= NUM_NODES

    encodings_merged = [None for _ in range(N)]
    node_idx_merged = node_idc_unique
    seq_length_merged = np.zeros(shape=N, dtype=np.int32)

    merged_idx_map = {v:i for i,v in enumerate(node_idx_merged)}
    to_merge = dict()
    for encodings, node_index, seq_length in encoding_sets:
        # assume that non-filled values are zero
        for i in range(node_index.shape[0]):
            idx = node_index[i]
            merged_idx = merged_idx_map[idx]

            enc = encodings[i]
            enc_length = seq_length[i]
            if idx in node_idc_mult:
                if idx not in to_merge:
                    to_merge[idx] = list()
                to_merge[idx].append((enc, enc_length))

            encodings_merged[merged_idx] = enc
            seq_length_merged[merged_idx] = enc_length

    for idx, encodings_and_lengths in to_merge.items():
        encodings, encodings_length = zip(*encodings_and_lengths)
        enc_length = max(encodings_length)
        enc_shape = max([enc.shape for enc in encodings])

        for i, enc in enumerate(encodings):
            if not isinstance(enc, sp.coo.coo_matrix):
                encodings[i] = enc.tocoo()

        n = len(encodings)
        data = np.concatenate([enc.data for enc in encodings]) / n
        row = np.concatenate([enc.row for enc in encodings])
        col = np.concatenate([enc.col for enc in encodings])
        enc = sp.coo_matrix((data, (row, col)), shape=enc_shape)

        merged_idx = merged_idx_map[idx]
        encodings_merged[merged_idx] = enc
        seq_length_merged[merged_idx] = enc_length

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
    merged_idx_multvalue_map = {merged_idx_map[v]:None for v in node_idc_mult}
    for encodings, node_index, seq_length in encoding_sets:
        # assume that non-filled values are zero
        for i in range(node_index.shape[0]):
            idx = node_index[i]
            merged_idx = merged_idx_map[idx]

            enc = encodings[i]
            if idx in node_idc_mult:
                mult_enc = merged_idx_multvalue_map[merged_idx]
                if mult_enc is None:
                    mult_enc = np.zeros(enc.shape, dtype=np.float32)
                merged_idx_multvalue_map[merged_idx] = mult_enc + enc

            encodings_merged[merged_idx,:encodings.shape[1]] = enc
            if seq_length is not None:
                # use max length for nodes that occur more than once
                seq_length_merged[merged_idx] = max(seq_length_merged[merged_idx],
                                                    seq_length[i])

    for i in range(node_idc_mult.shape[0]):
        idx = node_idc_mult[i]
        merged_idx = merged_idx_map[idx]
        count = node_idc_counts[node_idc_counts > 1][i]

        # average vectors for nodes that occur more than once
        encodings_merged[merged_idx] = merged_idx_multvalue_map[merged_idx] / count

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
    merged_idx_multvalue_map = {merged_idx_map[v]:None for v in node_idc_mult}
    for encodings, node_index, _ in encoding_sets:
        # assume that non-filled values are zero
        for i in range(node_index.shape[0]):
            idx = node_index[i]
            merged_idx = merged_idx_map[idx]

            enc = encodings[i]
            if idx in node_idc_mult:
                mult_enc = merged_idx_multvalue_map[merged_idx]
                if mult_enc is None:
                    mult_enc = np.zeros(enc.shape, dtype=np.float32)
                merged_idx_multvalue_map[merged_idx] = mult_enc + enc

            encodings_merged[merged_idx] = enc

    for i in range(node_idc_mult.shape[0]):
        idx = node_idc_mult[i]
        merged_idx = merged_idx_map[idx]
        count = node_idc_counts[node_idc_counts > 1][i]

        # average vectors for nodes that occur more than once
        encodings_merged[merged_idx] = merged_idx_multvalue_map[merged_idx] / count

    return [[encodings_merged, node_idx_merged, -np.ones(N)]]

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
