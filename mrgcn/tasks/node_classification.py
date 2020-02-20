#!/usr/bin/python3

import logging
from time import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim

from mrgcn.encodings.graph_features import construct_features
from mrgcn.models.mrgcn import MRGCN


logger = logging.getLogger(__name__)

def run(A, X, Y, C, tsv_writer, device, config,
        modules_config, featureless):
    tsv_writer.writerow(["epoch", "training_loss", "training_accurary",
                                  "validation_loss", "validation_accuracy",
                                  "test_loss", "test_accuracy"])

    # compile model
    model = build_model(C, Y, A, modules_config, config, featureless)
    optimizer = optim.Adam(model.parameters(),
                           lr=config['model']['learning_rate'],
                           weight_decay=config['model']['l2norm'])
    criterion = nn.CrossEntropyLoss()

    # mini batching
    mini_batch = config['model']['mini_batch']

    # early stopping
    patience = config['model']['patience']
    patience_left = patience
    best_score = -1
    delta = 1e-4
    best_state = None

    # train model
    nepoch = config['model']['epoch']
    # Log wall-clock time
    t0 = time()
    for epoch in train_model(A, model, optimizer, criterion, X, Y,
                             nepoch, mini_batch, device):
        # log metrics
        tsv_writer.writerow([str(epoch[0]),
                             str(epoch[1]),
                             str(epoch[2]),
                             str(epoch[3]),
                             str(epoch[4]),
                             "-1", "-1"])

        # early stopping
        val_loss = epoch[3]
        if patience <= 0:
            continue
        if best_score < 0:
            best_score = val_loss
            best_state = model.state_dict()
        if val_loss >= best_score - delta:
            patience_left -= 1
        else:
            best_score = val_loss
            best_state = model.state_dict()
            patience_left = patience
        if patience_left <= 0:
            model.load_state_dict(best_state)
            logger.info("Early stopping after no improvement for {} epoch".format(patience))
            break

    logging.info("Training time: {:.2f}s".format(time()-t0))

    # test model
    test_loss, test_acc = test_model(A, model, criterion, X, Y, device)
    # log metrics
    tsv_writer.writerow(["-1", "-1", "-1", "-1", "-1",
                         str(test_loss), str(test_acc)])

    return (test_loss, test_acc)

def train_model(A, model, optimizer, criterion, X, Y, nepoch, mini_batch, device):
    logging.info("Training for {} epoch".format(nepoch))
    for epoch in range(1, nepoch+1):
        batch_grad_idx = epoch - 1
        if not mini_batch:
            batch_grad_idx = -1

        # Single training iteration
        model.train()
        Y_hat = model(X, A,
                      batch_grad_idx=batch_grad_idx,
                      device=device)

        # Training scores
        train_loss = categorical_crossentropy(Y_hat, Y['train'], criterion)
        train_acc = categorical_accuracy(Y_hat, Y['train'])

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # validation scores
        model.eval()
        val_loss = categorical_crossentropy(Y_hat, Y['valid'], criterion)
        val_acc = categorical_accuracy(Y_hat, Y['valid'])

        # DEBUG #
        #for name, param in model.named_parameters():
        #    logger.info(name + " - grad mean: " + str(float(param.grad.mean())))
        # DEBUG #

        # cast criterion objects to floats to free the memory of the tensors
        # they point to
        train_loss = float(train_loss)
        train_acc = float(train_acc)
        val_loss = float(val_loss)
        val_acc = float(val_acc)

        logging.info("{:04d} ".format(epoch) \
                     + "| train loss {:.4f} / acc {:.4f} ".format(train_loss,
                                                                  train_acc)
                     + "| val loss {:.4f} / acc {:.4f}".format(val_loss,
                                                               val_acc))

        yield (epoch,
               train_loss, train_acc,
               val_loss, val_acc)

def test_model(A, model, criterion, X, Y, device):
    # Predict on full dataset
    model.train(False)
    with torch.no_grad():
        Y_hat = model(X, A,
                      batch_grad_idx=-1,
                      device=device)

    # scores on test set
    test_loss = categorical_crossentropy(Y_hat, Y['test'], criterion)
    test_acc = categorical_accuracy(Y_hat, Y['test'])

    test_loss = float(test_loss)
    test_acc = float(test_acc)

    logging.info("Performance on test set: loss {:.4f} / accuracy {:.4f}".format(
                  test_loss,
                  test_acc))

    return (test_loss, test_acc)

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
