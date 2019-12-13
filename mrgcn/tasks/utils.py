#!/usr/bin/env python

import logging

import numpy as np
from rdflib.term import URIRef
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
    if len(target_classes) > 0:
        for target_class in target_classes:
            # assume class is entity
            target_triples |= frozenset(knowledge_graph.triples((None,
                                                          URIRef(target_property),
                                                          URIRef(target_class))))
    else:
        target_triples |= frozenset(knowledge_graph.triples((None,
                                                      URIRef(target_property),
                                                      None)))

    knowledge_graph.graph -= target_triples  # strip targets from source

    # remove inverse target relations to prevent information leakage
    if target_property_inv != '':
        inv_target_triples = set()
        if len(target_classes) > 0:
            for target_class in target_classes:
                # assume class is entity
                inv_target_triples |= frozenset(knowledge_graph.triples((URIRef(target_class),
                                                                  URIRef(target_property_inv),
                                                                  None)))
        else:
            inv_target_triples |= frozenset(knowledge_graph.triples((None,
                                                              URIRef(target_property_inv),
                                                              None)))
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

def mkbatches(mat, node_idx, C, _, nbins=10, batch_size=-1):
    """ split N x * array in batches
    """
    n = mat.shape[0]  # number of samples
    idc = np.array(range(n), dtype=np.int32)
    if batch_size > 0:
        nbins = np.ceil(len(idc)/batch_size)
    idc_assignments = np.array_split(idc, nbins)

    node_assignments = [np.array(node_idx, dtype=np.int32)[slce]
                        for slce in idc_assignments]

    return list(zip(idc_assignments, node_assignments))

def mkbatches_varlength(sequences, node_idx, C, seq_length_map, max_bins=-1):
    """ :param sequences: a list with M arrays of length ?
                    M :- number of nodes with this feature M <= N
        :param node_idx: list that maps sequence idx {0, M} to node idx {0, N}
        :param seq_length_map: list that maps sequence idx {0, M} to length {0, K}
        :returns: list with B lists, each holding sequence indices for that batch;
                  list with B lists, each holding node indices for sequences;

    """
    if max_bins <= 0:
        max_bins = 16

    # split outliers
    q25 = np.quantile(seq_length_map, 0.25)
    q75 = np.quantile(seq_length_map, 0.75)
    IQR = q75 - q25
    cut_off = IQR * 1.5

    outliers_low = list()
    outliers_high = list()
    non_outliers = list()
    for seq_length in seq_length_map:
        if seq_length < q25 - cut_off:
            outliers_low.append(seq_length)
            continue
        if seq_length > q75 + cut_off:
            outliers_high.append(seq_length)
            continue

        non_outliers.append(seq_length)

    # determine optimal number of bins using the Freedman-Diaconis rule
    h = 2 * IQR / np.power(len(non_outliers), 1/3)
    nbins = min(max_bins,
                np.round((max(non_outliers)-min(non_outliers)) / h))

    # create bins
    bin_ranges = np.array_split(np.unique(non_outliers), nbins)
    for outlier_bin in [outliers_low, outliers_high]:
        if len(outlier_bin) <= 0:
            continue
        bin_ranges.append(np.array(outlier_bin))

    bin_ranges_map = {length:bin_idx for bin_idx in range(len(bin_ranges))
                      for length in bin_ranges[bin_idx]}

    # assign sequences
    seq_assignments = [list() for bin_range in bin_ranges]
    node_assignments = [list() for bin_range in bin_ranges]
    for i in range(len(sequences)):
        length = seq_length_map[i]
        seq_assignments[bin_ranges_map[length]].append(i)
        node_assignments[bin_ranges_map[length]].append(node_idx[i])

    return list(zip(seq_assignments, node_assignments))

