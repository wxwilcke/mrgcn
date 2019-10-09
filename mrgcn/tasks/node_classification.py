#!/usr/bin/python3

import logging

import numpy as np
import torch
import torch.nn as nn

from mrgcn.embeddings.graph_features import construct_features
from mrgcn.models.mrgcn import MRGCN


logger = logging.getLogger(__name__)

def generate_task(knowledge_graph, A, targets, config):
    logger.debug("Generating node classification task")
    X, Y, X_node_idx = build_dataset(knowledge_graph, targets, config)
    model = build_model(X,
                        Y,
                        A, config)

    return (X, Y, X_node_idx, model)

def build_dataset(knowledge_graph, target_triples, config, featureless):
    logger.debug("Starting dataset build")
    # generate target matrix
    classes = {t[2] for t in target_triples}  # unique classes
    logger.debug("Found {} instances (statements)".format(len(target_triples)))
    logger.debug("Target classes ({}): {}".format(len(classes), classes))

    # node/class label to integers
    nodes_map = {label:i for i,label in enumerate(knowledge_graph.atoms())}
    classes_map = {label:i for i,label in enumerate(classes)}
    num_nodes = len(nodes_map)
    num_classes = len(classes_map)

    target_indices = [(nodes_map[x], classes_map[y]) for x, _, y in target_triples]
    X_node_idx, Y_class_idx = map(np.array, zip(*target_indices))

    # matrix of 1-hot class vectors per node
    Y = np.zeros((num_nodes, num_classes), dtype=np.int8)
    for i,j in zip(X_node_idx, Y_class_idx):
        Y[i, j] = 1

    if featureless:
        # use uninitialized matrix when featureless
        X = None
    else:
        X = construct_features(nodes_map, config['graph']['features'])

    logger.debug("Completed dataset build")
    return (X, Y, X_node_idx)

def build_model(X, Y, A, config, featureless):
    layers = config['model']['layers']
    assert len(layers) >= 2
    logger.debug("Starting model build")

    # should be pyTorch tensors
    num_nodes, X_dim = X.size()
    num_relations = int(A.size()[1]/num_nodes)
    Y_dim = Y.size()[1]

    modules = list()
    # input layer
    modules.append((X_dim,
                    layers[0]['hidden_nodes'],
                    nn.ReLU()))

    # intermediate layers (if any)
    i = 1
    for layer in layers[1:-1]:
        modules.append((layers[i-1]['hidden_nodes'],
                        layer['hidden_nodes'],
                        nn.ReLU()))

        i += 1

    # output layer
    # applies softmax over possible classes
    modules.append((layers[i-1]['hidden_nodes'],
                    Y_dim,
                    nn.Softmax(dim=1)))

    model = MRGCN(modules, num_relations, num_nodes,
                  num_bases=config['model']['num_bases'],
                  p_dropout=config['model']['p_dropout'],
                  featureless=featureless,
                  bias=config['model']['bias'])

    logger.debug("Completed model build")

    return model

def categorical_accuracy(Y_hat, Y, idx):
    _, labels = Y_hat[idx].max(dim=1)
    _, targets = Y[idx].max(dim=1)

    return torch.mean(torch.eq(labels, targets).float())

def categorical_crossentropy(Y_hat, Y, idx, criterion):
    predictions = Y_hat[idx]
    _, targets = Y[idx].max(dim=1)

    return criterion(predictions, targets)
