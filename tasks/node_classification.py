#!/usr/bin/python3

import logging

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np
import scipy.sparse as sp

from layers.graph import GraphConvolution
from layers.input_adj import InputAdj


logger = logging.getLogger(__name__)

def generate_task(knowledge_graph, A, targets, config):
    logger.debug("Generating node classification task")
    X, Y, X_node_idx = build_dataset(knowledge_graph, targets, config)
    model = build_model(X,
                        Y,
                        A, config)

    return (X, Y, X_node_idx, model)

def build_dataset(knowledge_graph, target_triples, config):
    logger.debug("Starting dataset build")
    # generate target matrix
    classes = {t[2] for t in target_triples}  # unique classes
    logger.debug("Found {} instances (statements)".format(len(target_triples)))
    logger.debug("Target classes ({}): {}".format(len(classes), classes))
    
    nodes_map = {label:i for i,label in enumerate(knowledge_graph.atoms())}
    classes_map = {label:i for i,label in enumerate(classes)}

    # matrix of 1-hot class vectors per node
    target_labels = [(nodes_map[x], classes_map[y]) for x, _, y in target_triples]
    X_node_idx, Y_class_idx = map(np.array, zip(*target_labels))
    Y = sp.csr_matrix((np.ones(len(target_labels)), (X_node_idx, Y_class_idx)),
                               shape=(len(nodes_map), len(classes_map)),
                                 dtype=np.int32)
    
    # dummy matrix for featureless learning
    X = sp.csr_matrix((len(nodes_map), len(nodes_map)))

    return (X, Y, X_node_idx)

def build_model(X, Y, A, config):
    layers = config['model']['layers']
    assert len(layers) >= 2
    logger.debug("Starting model build")

    support = len(A)
    A_in = [InputAdj(sparse=True) for _ in range(support)]

    # input layer
    X_in = Input(shape=(X.shape[1],), sparse=True)
    H = GraphConvolution(layers[0]['hidden_nodes'], 
                         support, 
                         num_bases=layers[0]['num_bases'],
                         featureless=layers[0]['featureless'],
                         activation=layers[0]['activation'],
                         W_regularizer=l2(layers[0]['l2norm']))([X_in] + A_in)
    H = Dropout(layers[0]['dropout'])(H)

    # intermediate layers (if any)
    for i, layer in enumerate(layers[1:-1], 1):
        H = GraphConvolution(layers[i]['hidden_nodes'], 
                             support, 
                             num_bases=layers[i]['num_bases'],
                             featureless=layers[i]['featureless'],
                             activation=layers[i]['activation'],
                             W_regularizer=l2(layers[i]['l2norm']))([H] + A_in)
        H = Dropout(layers[i]['dropout'])(H)

    # output layer
    Y_out = GraphConvolution(Y.shape[1], support, 
                             num_bases=layers[-1]['num_bases'],
                             activation=layers[-1]['activation'])([H] + A_in)

    # Compile model
    logger.debug("Compiling model")
    model = Model(inputs=[X_in] + A_in, outputs=Y_out)
    model.compile(loss=config['model']['loss'],
                  optimizer=Adam(lr=config['model']['learning_rate']))

    return model

def evaluate_model(Y_hat, Y, indices):
    split_loss = []
    split_acc = []

    for Y_split, idx_split in zip(Y, indices):
        split_loss.append(categorical_crossentropy(Y_hat[idx_split], Y_split[idx_split]))
        split_acc.append(categorical_accuracy(Y_hat[idx_split], Y_split[idx_split]))

    return split_loss, split_acc

def categorical_accuracy(Y_hat, Y):
    return np.mean(np.equal(np.argmax(Y, 1), np.argmax(Y_hat, 1)))

def categorical_crossentropy(Y_hat, Y):
    return np.mean(-np.log(np.extract(Y, Y_hat)))
