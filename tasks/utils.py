#!/usr/bin/python3

import logging
import os
import random

import numpy as np
from rdflib.term import URIRef

logger = logging.getLogger(__name__)

def set_seed(seed=-1):
    # Note: this does not work well somehow
    if seed >= 0:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

        logger.info("Setting seed to {}".format(seed))
    else:
        logger.info("Using random seed")

def strip_graph(knowledge_graph, config):
    target_property = config['task']['target_property'] 
    target_property_inv = config['task']['target_property_inv'] 
    #target_classes = config['task']['target_classes']
    
    n = knowledge_graph.__len__()
    logger.info("Stripping knowledge graph...")
    # list of target triples (entity-class mapping)
    target_triples = list(knowledge_graph.triples(URIRef(target_property)))
    knowledge_graph.graph -= target_triples  # strip targets from source

    # remove inverse target relations to prevent information leakage
    if target_property_inv != '':
        knowledge_graph.graph -= list(knowledge_graph.triples(
                                        URIRef(target_property_inv)))

    m = knowledge_graph.__len__()
    logger.info("Stripped {} statements ({} remain)".format(n-m, m))

    return target_triples

def create_splits(X, Y, X_nodes_map, dataset_ratio=(.6,.2,.2), shuffle=True): 
    assert sum(dataset_ratio) == 1.0
    logger.info("Creating train-test-validation sets with ratio {}".format(dataset_ratio))
    # indices of targets
    idx = np.array(range(X_nodes_map.shape[0]))
    if shuffle:
        logger.info("Shuffling dataset")
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

def sample_mask(idx, l):
    logger.info("Adding sample mask")
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
