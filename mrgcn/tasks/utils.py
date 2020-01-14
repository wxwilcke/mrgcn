#!/usr/bin/env python

import logging

import numpy as np
from rdflib.term import Literal, URIRef
import torch


logger = logging.getLogger(__name__)

def strip_graph(knowledge_graph, config):
    target_property = config['task']['target_property']
    target_property_inv = config['task']['target_property_inv']
    target_classes = config['task']['target_classes']

    n = len(knowledge_graph)
    logger.debug("Stripping knowledge graph...")
    # list of target triples (entity-class mapping)
    target_triples = set()
    separate_literals = config['graph']['structural']['separate_literals']
    if len(target_classes) > 0:
        for target_class in target_classes:
            # assume class is entity
            target_triples |= frozenset(knowledge_graph.triples((None,
                                                          URIRef(target_property),
                                                          URIRef(target_class)),
                                                        separate_literals))
    else:
        target_triples |= frozenset(knowledge_graph.triples((None,
                                                      URIRef(target_property),
                                                      None),
                                                    separate_literals))

    if not separate_literals:
        knowledge_graph.graph -= target_triples  # strip targets from source
    else:
        for s, p, o in target_triples:
            o_source = Literal(str(o), o.language, o.datatype, normalize=None)\
                    if type(o) is Literal else o
            knowledge_graph.graph.remove((s, p, o_source))

    # remove inverse target relations to prevent information leakage
    if target_property_inv != '':
        inv_target_triples = set()
        if len(target_classes) > 0:
            for target_class in target_classes:
                # assume class is entity
                inv_target_triples |= frozenset(knowledge_graph.triples((URIRef(target_class),
                                                                  URIRef(target_property_inv),
                                                                  None),
                                                                separate_literals))
        else:
            inv_target_triples |= frozenset(knowledge_graph.triples((None,
                                                              URIRef(target_property_inv),
                                                              None),
                                                            separate_literals))
        knowledge_graph.graph -= inv_target_triples

    m = len(knowledge_graph)
    logger.debug("Stripped {} statements ({} remain)".format(n-m, m))

    return target_triples

def mkfolds(n, k, shuffle=True):
    assert k > 2
    logger.debug("Generating {} folds".format(k))

    # indices of targets
    idx = np.array(range(n))
    if shuffle:
        logger.debug("Shuffling dataset")
        np.random.shuffle(idx)

    fold_size = int(len(idx)/k)

    folds = []
    for i in range(k):
        test_idx = idx[fold_size*i:fold_size*(i+1)]
        train_idx = np.setdiff1d(idx, test_idx)  # takes remainder
        folds.append({'train':train_idx, 'test':test_idx})

    return folds

def init_fold(X, Y, X_nodes_map, idx_dict, device, dataset_ratio=(.7,.2,.1)):
    assert round(sum(dataset_ratio), 2) == 1.0
    logger.debug("Initializing fold")

    # calculate validation split
    val_split = dataset_ratio[-1]
    val_idx = int(len(idx_dict['train'])*val_split)

    # split datasets
    X_train_idx = torch.LongTensor(X_nodes_map[idx_dict['train'][:-val_idx]])
    X_test_idx = torch.LongTensor(X_nodes_map[idx_dict['test']])
    X_val_idx = torch.LongTensor(X_nodes_map[idx_dict['train'][-val_idx:]])

    # split Y
    Y_train = torch.zeros(Y.size())
    Y_test = torch.zeros(Y.size())
    Y_val = torch.zeros(Y.size())

    Y_train[X_train_idx] = Y[X_train_idx].float()
    Y_test[X_test_idx] = Y[X_test_idx].float()
    Y_val[X_val_idx] = Y[X_val_idx].float()

    # X stays unmodified 
    # X = X

    return { 'train': { 'X': X, 'Y': Y_train, 'idx': X_train_idx },
            'test': { 'X': X, 'Y': Y_test, 'idx': X_test_idx },
            'val': { 'X': X, 'Y': Y_val, 'idx': X_val_idx }}

def mksplits(X, Y, X_nodes_map, device, dataset_ratio=(.7,.2,.1), shuffle=True):
    assert round(sum(dataset_ratio), 2) == 1.0
    logger.debug("Creating train-test-validation sets with ratio {}".format(dataset_ratio))
    # indices of targets
    idx = np.array(range(X_nodes_map.shape[0]))
    if shuffle:
        logger.debug("Shuffling dataset")
        np.random.shuffle(idx)

    # create splits
    train_idx = idx[:int(dataset_ratio[0]*len(idx))]
    test_idx = idx[len(train_idx):len(train_idx)+int(dataset_ratio[1]*len(idx))]
    val_idx = idx[len(train_idx)+len(test_idx):]

    # split datasets
    X_train_idx = torch.LongTensor(X_nodes_map[train_idx])
    X_test_idx = torch.LongTensor(X_nodes_map[test_idx])
    X_val_idx = torch.LongTensor(X_nodes_map[val_idx])

    # split Y
    Y_train = torch.zeros(Y.size())
    Y_test = torch.zeros(Y.size())
    Y_val = torch.zeros(Y.size())

    Y_train[X_train_idx] = Y[X_train_idx].float()
    Y_test[X_test_idx] = Y[X_test_idx].float()
    Y_val[X_val_idx] = Y[X_val_idx].float()

    # X stays unmodified 
    # X = X

    return { 'train': { 'X': X, 'Y': Y_train, 'idx': X_train_idx },
            'test': { 'X': X, 'Y': Y_test, 'idx': X_test_idx },
            'val': { 'X': X, 'Y': Y_val, 'idx': X_val_idx }}

def dataset_to_device(dataset, device):
    for split in dataset.values():
        split['Y'] = split['Y'].to(device)
        split['idx'] = split['idx'].to(device)
        # X stays where it is

def sample_mask(idx, n):
    mask = np.zeros(n)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def mkbatches(mat, node_idx, C, _, nsets, nepoch, passes_per_batch=1):
    """ split N x * array in batches
    """
    n = mat.shape[0]  # number of samples
    idc = np.array(range(n), dtype=np.int32)

    nbins = nepoch
    if passes_per_batch > 1:
        nbins = nepoch//passes_per_batch

    idc_assignments = np.array_split(idc, nbins)
    node_assignments = [np.array(node_idx, dtype=np.int32)[slce]
                        for slce in idc_assignments]

    return list(zip(idc_assignments, node_assignments))

def mkbatches_varlength(sequences, node_idx, C, seq_length_map, _,
                            nepoch, passes_per_batch=1):
    n = len(sequences)
    # sort on length
    idc = np.array(range(n), dtype=np.int32)
    seq_length_map_sorted, node_idx_sorted, sequences_sorted_idc = zip(
        *sorted(zip(seq_length_map,
                    node_idx,
                    idc)))

    nbins = nepoch
    if passes_per_batch > 1:
        nbins = nepoch//passes_per_batch

    seq_assignments = np.array_split(sequences_sorted_idc, nbins)
    node_assignments = [np.array(node_idx, dtype=np.int32)[slce]
                        for slce in seq_assignments]

    return list(zip(seq_assignments, node_assignments))

def remove_outliers(sequences, node_idx, C, seq_length_map, nsets):
    # split outliers
    q25 = np.quantile(seq_length_map, 0.25)
    q75 = np.quantile(seq_length_map, 0.75)
    IQR = q75 - q25
    cut_off = IQR * 1.5

    if IQR <= 0.0:  # no length difference
        return [sequences, node_idx, C, seq_length_map, nsets]

    sequences_filtered = list()
    node_idx_filtered = list()
    seq_length_map_filtered = list()
    for i, seq_length in enumerate(seq_length_map):
        if seq_length < q25 - cut_off or seq_length > q75 + cut_off:
            # skip outlier
            continue

        sequences_filtered.append(sequences[i])
        node_idx_filtered.append(node_idx[i])
        seq_length_map_filtered.append(seq_length)

    n = len(sequences_filtered)
    d = len(sequences) - n
    if d > 0:
        logger.debug("Filtered {} outliers ({} remain)".format(d, n))

    return [sequences_filtered, node_idx_filtered, C, seq_length_map_filtered, nsets]
