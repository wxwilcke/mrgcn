#!/usr/bin/env python

import logging
import os
import random

from keras import backend as K
import numpy as np
from rdflib.term import URIRef
import tensorflow as tf

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

def init_fold(X, Y, X_nodes_map, idx_dict, dataset_ratio=(.7,.2,.1)):
    assert round(sum(dataset_ratio), 2) == 1.0
    logger.debug("Initializing fold")

    # calculate validation split
    val_split = dataset_ratio[-1]
    val_idx = int(len(idx_dict['train'])*val_split)

    # split datasets
    X_train_idx = X_nodes_map[idx_dict['train'][:-val_idx]]
    X_test_idx = X_nodes_map[idx_dict['test']]
    X_val_idx = X_nodes_map[idx_dict['train'][-val_idx:]]

    # split Y
    Y_train = np.zeros(Y.shape)
    Y_test = np.zeros(Y.shape)
    Y_val = np.zeros(Y.shape)

    Y_train[X_train_idx] = np.array(Y[X_train_idx].todense())
    Y_test[X_test_idx] = np.array(Y[X_test_idx].todense())
    Y_val[X_val_idx] = np.array(Y[X_val_idx].todense())

    # X stays unmodified during featureless learning
    # X = X

    return { 'train': { 'X': X, 'Y': Y_train, 'X_idx': X_train_idx },
            'test': { 'X': X, 'Y': Y_test, 'X_idx': X_test_idx },
            'val': { 'X': X, 'Y': Y_val, 'X_idx': X_val_idx }}

def mksplits(X, Y, X_nodes_map, dataset_ratio=(.7,.2,.1), shuffle=True):
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
    X_train_idx = X_nodes_map[train_idx]
    X_test_idx = X_nodes_map[test_idx]
    X_val_idx = X_nodes_map[val_idx]

    # split Y
    Y_train = np.zeros(Y.shape)
    Y_test = np.zeros(Y.shape)
    Y_val = np.zeros(Y.shape)

    Y_train[X_train_idx] = np.array(Y[X_train_idx].todense())
    Y_test[X_test_idx] = np.array(Y[X_test_idx].todense())
    Y_val[X_val_idx] = np.array(Y[X_val_idx].todense())

    # X stays unmodified during featureless learning
    # X = X

    return { 'train': { 'X': X, 'Y': Y_train, 'X_idx': X_train_idx },
            'test': { 'X': X, 'Y': Y_test, 'X_idx': X_test_idx },
            'val': { 'X': X, 'Y': Y_val, 'X_idx': X_val_idx }}

def sample_mask(idx, n):
    mask = np.zeros(n)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
