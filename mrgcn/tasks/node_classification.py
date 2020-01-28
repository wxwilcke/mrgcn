#!/usr/bin/python3

import logging

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from mrgcn.encodings.graph_features import construct_features
from mrgcn.models.mrgcn import MRGCN


logger = logging.getLogger(__name__)

def build_dataset(knowledge_graph, nodes_map, target_triples, config, featureless):
    logger.debug("Starting dataset build")
    # generate target matrix
    Y = mk_target_matrices(target_triples, nodes_map)

    if featureless:
        F = dict()
    else:
        separate_literals = config['graph']['structural']['separate_literals']
        F = construct_features(nodes_map, knowledge_graph,
                               config['graph']['features'],
                               separate_literals)

    logger.debug("Completed dataset build")

    return (F, Y)

def mk_target_matrices(target_triples, nodes_map):
    classes = {str(c) for split in target_triples.values() for _,_,c in split} # unique classes
    logger.debug("Target classes ({}): {}".format(len(classes), classes))

    # node/class label to integers
    classes_map = {label:i for i,label in enumerate(classes)}

    # note: by converting targets to strings we lose datatype info, but the use
    # cases where this would matter would be very limited 
    num_nodes = len(nodes_map)
    num_classes = len(classes_map)
    Y = dict()
    for k, split in target_triples.items():
        logger.debug("Found {} instances ({})".format(len(split), k))
        target_pair_indices = [(nodes_map[x], classes_map[str(y)]) for x, _, y in split]
        rows, cols = map(np.array, zip(*target_pair_indices))
        data = np.ones(len(rows), dtype=np.int8)
        Y[k] = sp.csr_matrix((data, (rows, cols)),
                             shape=(num_nodes, num_classes),
                             dtype=np.int8)

    return Y

def build_model(C, Y, A, modules_config, config, featureless):
    layers = config['model']['layers']
    assert len(layers) >= 2
    logger.debug("Starting model build")

    # get sizes from dataset
    X_dim = C  # == 0 if featureless
    num_nodes, Y_dim = Y['train'].shape
    num_relations = int(A.size()[1]/num_nodes)

    modules = list()
    # input layer
    modules.append((X_dim,
                    layers[0]['hidden_nodes'],
                    layers[0]['type'],
                    nn.ReLU()))

    # intermediate layers (if any)
    i = 1
    for layer in layers[1:-1]:
        modules.append((layers[i-1]['hidden_nodes'],
                        layer['hidden_nodes'],
                        layers[i-1]['type'],
                        nn.ReLU()))

        i += 1

    # output layer
    # applies softmax over possible classes
    modules.append((layers[i-1]['hidden_nodes'],
                    Y_dim,
                    layers[i-1]['type'],
                    nn.Softmax(dim=1)))

    model = MRGCN(modules, modules_config, num_relations, num_nodes,
                  num_bases=config['model']['num_bases'],
                  p_dropout=config['model']['p_dropout'],
                  featureless=featureless,
                  bias=config['model']['bias'])

    logger.debug("Completed model build")

    return model

def categorical_accuracy(Y_hat, Y):
    idx, targets = Y.nonzero()
    targets = torch.as_tensor(targets, dtype=torch.long)
    _, labels = Y_hat[idx].max(dim=1)

    return torch.mean(torch.eq(labels, targets).float())

def categorical_crossentropy(Y_hat, Y, criterion):
    idx, targets = Y.nonzero()
    targets = torch.as_tensor(targets, dtype=torch.long)
    predictions = Y_hat[idx]

    return criterion(predictions, targets)
