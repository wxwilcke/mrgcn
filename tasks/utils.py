#!/usr/bin/python3

import logging
from random import seed as set_seed
from random import shuffle as shuffler

import numpy as np
from rdflib.term import URIRef

logger = logging.getLogger(__name__)

def strip_graph(knowledge_graph, config):
    target_property = config['task']['target_property'] 
    target_property_inv = config['task']['target_property_inv'] 
    #target_classes = config['task']['target_classes']
       
    # list of target triples (entity-class mapping)
    target_triples = list(knowledge_graph.triples(URIRef(target_property)))
    knowledge_graph.graph -= target_triples  # strip targets from source
   
    # remove inverse target relations to prevent information leakage
    if target_property_inv != '':
        knowledge_graph.graph -= list(knowledge_graph.triples(
                                        URIRef(target_property_inv)))

    return target_triples

def create_splits(X, Y, X_nodes_map, dataset_ratio=(.6,.2,.2), shuffle=True, seed=-1): 
    assert sum(dataset_ratio) == 1.0
    logger.info("Set train-test-validation sets to ratio {}".format(dataset_ratio))
    if seed >= 0:
        set_seed(seed)
        logger.info("Set seed to {}".format(seed))

    # indices of targets
    idx = np.array(range(X_nodes_map.shape[0]))
    if shuffle:
        shuffler(idx)

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
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
