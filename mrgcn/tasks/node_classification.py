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
from mrgcn.tasks.utils import optimizer_params


logger = logging.getLogger()


def run(A, X, Y, X_width, tsv_writer, device, config,
        modules_config, optimizer_config, featureless, test_split):
    tsv_writer.writerow(["epoch", "training_loss", "training_accurary",
                                  "validation_loss", "validation_accuracy",
                                  "test_loss", "test_accuracy"])

    # compile model
    model = build_model(X_width, Y, A, modules_config, config, featureless)
    opt_params = optimizer_params(model, optimizer_config)
    optimizer = optim.Adam(opt_params,
                           lr=config['model']['learning_rate'],
                           weight_decay=config['model']['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    # train model
    nepoch = config['model']['epoch']
    l1_lambda = config['model']['l1_lambda']
    l2_lambda = config['model']['l2_lambda']
    # Log wall-clock time
    t0 = time()
    for epoch in train_model(A, model, optimizer, criterion, X, Y,
                             nepoch, test_split, l1_lambda, l2_lambda, device):
        # log metrics
        tsv_writer.writerow([str(epoch[0]),
                             str(epoch[1]),
                             str(epoch[2]),
                             str(epoch[3]),
                             str(epoch[4]),
                             "-1", "-1"])

    logging.info("Training time: {:.2f}s".format(time()-t0))

    # test model
    loss, acc, labels, targets = test_model(A, model, criterion, X, Y, test_split, device)
    # log metrics
    tsv_writer.writerow(["-1", "-1", "-1", "-1", "-1",
                         str(loss), str(acc)])

    return (loss, acc, labels, targets)


def train_model(A, model, optimizer, criterion, X, Y, nepoch, test_split,
                l1_lambda, l2_lambda, device):
    Y_train = Y['train']
    Y_valid = Y['valid']
    if test_split == "test":
        # merge training and validation sets
        ri = np.concatenate([Y_train.nonzero()[0], Y_valid.nonzero()[0]])
        ci = np.concatenate([Y_train.nonzero()[1], Y_valid.nonzero()[1]])
        d = np.concatenate([Y_train.data, Y_valid.data])

        Y_train = sp.csr_matrix((d, (ri, ci)))
        Y_valid = None

    logging.info("Training for {} epoch".format(nepoch))
    for epoch in range(1, nepoch+1):
        # Single training iteration
        model.train()

        optimizer.zero_grad()
        # Training scores
        Y_hat = model(X, A, epoch, device=device).to('cpu')
        train_loss = categorical_crossentropy(Y_hat, Y_train, criterion)
        train_acc = categorical_accuracy(Y_hat, Y_train)[0]

        if l1_lambda > 0:
            l1_regularization = torch.tensor(0.)
            for name, param in model.named_parameters():
                if 'weight' not in name:
                    continue
                l1_regularization += torch.sum(param.abs())

            train_loss += l1_lambda * l1_regularization

        if l2_lambda > 0:
            l2_regularization = torch.tensor(0.)
            for name, param in model.named_parameters():
                if 'weight' not in name:
                    continue
                l2_regularization += torch.sum(param ** 2)

            train_loss += l2_lambda * l2_regularization

        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # cast criterion objects to floats to free the memory of the tensors
        # they point to
        train_loss = float(train_loss)
        train_acc = float(train_acc)

        val_loss = -1
        val_acc = -1
        if Y_valid is not None:
            # validation scores
            model.eval()

            with torch.no_grad():
                val_loss = categorical_crossentropy(Y_hat, Y_valid, criterion)
                val_acc = categorical_accuracy(Y_hat, Y_valid)[0]

            # DEBUG #
            #for name, param in model.named_parameters():
            #    logger.info(name + " - grad mean: " + str(float(param.grad.mean())))
            # DEBUG #
            val_loss = float(val_loss)
            val_acc = float(val_acc)

            logging.info("{:04d} ".format(epoch) \
                         + "| train loss {:.4f} / acc {:.4f} ".format(train_loss,
                                                                      train_acc)
                         + "| val loss {:.4f} / acc {:.4f}".format(val_loss,
                                                                   val_acc))
        else:
            logging.info("{:04d} ".format(epoch) \
                         + "| train loss {:.4f} / acc {:.4f} ".format(train_loss,
                                                                      train_acc))

        yield (epoch,
               train_loss, train_acc,
               val_loss, val_acc)

def test_model(A, model, criterion, X, Y, test_split, device):
    # Predict on full dataset
    model.train(False)
    with torch.no_grad():
        Y_hat = model(X, A, epoch=-1, device=device).to('cpu')

    # scores on set
    loss = categorical_crossentropy(Y_hat, Y[test_split], criterion)
    acc, labels, targets = categorical_accuracy(Y_hat, Y[test_split])

    loss = float(loss)
    acc = float(acc)

    logging.info("Performance on {} set: loss {:.4f} / accuracy {:.4f}".format(
                  test_split,
                  loss,
                  acc))

    return (loss, acc, labels, targets)

def build_dataset(knowledge_graph, nodes_map, target_triples, config, featureless):
    logger.debug("Starting dataset build")
    # generate target matrix
    Y, sample_map, class_map = mk_target_matrices(target_triples, nodes_map)

    if featureless:
        F = dict()
    else:
        separate_literals = config['graph']['structural']['separate_literals']
        F = construct_features(nodes_map, knowledge_graph,
                               config['graph']['features'],
                               separate_literals)

    logger.debug("Completed dataset build")

    return (F, Y, sample_map, class_map)

def mk_target_matrices(target_triples, nodes_map):
    classes = {str(c) for split in target_triples.values() for _,_,c in split} # unique classes
    logger.debug("Target classes ({}): {}".format(len(classes), classes))

    # node/class label to integers; sorted to make consistent over runs
    class_map = sorted(list(classes))
    class_map_inv = {label:i for i,label in enumerate(class_map)}

    # note: by converting targets to strings we lose datatype info, but the use
    # cases where this would matter would be very limited 
    num_nodes = len(nodes_map)
    num_classes = len(class_map)
    sample_map = dict()
    Y = dict()
    for k in target_triples.keys():
        split = sorted(target_triples[k])

        logger.debug("Found {} instances ({})".format(len(split), k))
        target_pair_indices = list()
        sample_map[k] = list()
        for x, _, y in split:
            target_pair_indices.append((nodes_map[x], class_map_inv[str(y)]))
            sample_map[k].append(x)

        rows, cols = map(np.array, zip(*target_pair_indices))
        data = np.ones(len(rows), dtype=np.int8)  # assume <= 128 classes
        Y[k] = sp.csr_matrix((data, (rows, cols)),
                             shape=(num_nodes, num_classes),
                             dtype=np.int8)

    return (Y, sample_map, class_map)

def build_model(X_width, Y, A, modules_config, config, featureless):
    layers = config['model']['layers']
    assert len(layers) >= 2
    logger.debug("Starting model build")

    # get sizes from dataset
    num_nodes, Y_dim = Y['train'].shape
    num_relations = int(A.shape[1]/num_nodes)

    modules = list()
    # input layer
    modules.append((X_width,
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
    modules.append((layers[i-1]['hidden_nodes'],
                    Y_dim,
                    layers[i-1]['type'],
                    None))

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

    return (torch.mean(torch.eq(labels, targets).float()), labels, targets)

def categorical_crossentropy(Y_hat, Y, criterion):
    idx, targets = Y.nonzero()
    targets = torch.as_tensor(targets, dtype=torch.long)
    predictions = Y_hat[idx]

    return criterion(predictions, targets)
