#!/usr/bin/python3

import logging
import argparse
from os import getpid
from time import time

import numpy as np
import toml
import torch
import torch.nn as nn
import torch.optim as optim

from mrgcn.data.io.knowledge_graph import KnowledgeGraph
from mrgcn.data.io.tarball import Tarball
from mrgcn.data.io.tsv import TSV
from mrgcn.data.utils import is_readable, is_writable
from mrgcn.embeddings import graph_structure
from mrgcn.tasks.config import set_seed
from mrgcn.data.utils import scipy_sparse_to_pytorch_sparse
from mrgcn.tasks.node_classification import (build_dataset,
                                             build_model,
                                             categorical_accuracy,
                                             categorical_crossentropy)
from mrgcn.tasks.utils import mksplits, init_fold, mkfolds, strip_graph


VERSION = 0.1

def single_run(A, X, Y, X_node_map, tsv_writer, device, config, featureless):
    tsv_writer.writerow(["epoch", "training_loss", "training_accurary",
                                  "validation_loss", "validation_accuracy",
                                  "test_loss", "test_accuracy"])

    # create splits
    dataset = mksplits(X, Y, X_node_map,
                            config['task']['dataset_ratio'])

    # compile model and move to gpu if possible
    model = build_model(X, Y, A, config, featureless)
    model.to(device, non_blocking=True)

    optimizer = optim.Adam(model.parameters(),
                           lr=config['model']['learning_rate'],
                           weight_decay=config['model']['l2norm'])
    criterion = nn.CrossEntropyLoss()

    # train model
    nepoch = config['model']['epoch']
    batch_size = X.size()[0]  # number of nodes

    for epoch in train_model(A, model, optimizer, criterion, dataset,
                             batch_size, nepoch):
        # log metrics
        tsv_writer.writerow([str(epoch[0]),
                             str(epoch[1]),
                             str(epoch[2]),
                             str(epoch[3]),
                             str(epoch[4]),
                             "-1", "-1"])

    # test model
    test_loss, test_acc = test_model(A, model, criterion, dataset)
    # log metrics
    tsv_writer.writerow(["-1", "-1", "-1", "-1", "-1",
                         str(test_loss), str(test_acc)])

    return (test_loss, test_acc)

def kfold_crossvalidation(A, X, Y, X_node_map, k, tsv_writer, device, config,
                          featureless):
    tsv_writer.writerow(["fold", "epoch",
                         "training_loss", "training_accurary",
                         "validation_loss", "validation_accuracy",
                         "test_loss", "test_accuracy"])

    # compile model and move to gpu if possible
    model = build_model(X, Y, A, config, featureless)
    model.to(device, non_blocking=True)

    optimizer = optim.Adam(model.parameters(),
                           lr=config['model']['learning_rate'],
                           weight_decay=config['model']['l2norm'])
    optimizer_state_zero = optimizer.state_dict()  # save initial state
    criterion = nn.CrossEntropyLoss()

    # generate fold indices
    folds_idx = mkfolds(X_node_map.shape[0], k)

    results = []
    logger.info("Starting {}-fold cross validation".format(k))
    for fold in range(1, k+1):
        logger.info("Fold {} / {}".format(fold, k))
        # initialize fold
        dataset = init_fold(X, Y, X_node_map, folds_idx[fold-1],
                            config['task']['dataset_ratio'])

        # train model
        nepoch = config['model']['epoch']
        batch_size = X.size()[0]  # number of nodes

        for epoch in train_model(A, model, optimizer, criterion, dataset,
                                 batch_size, nepoch):
            # log metrics
            tsv_writer.writerow([str(fold),
                                 str(epoch[0]),
                                 str(epoch[1]),
                                 str(epoch[2]),
                                 str(epoch[3]),
                                 str(epoch[4]),
                                 "-1", "-1"])

        # test model
        test_loss, test_acc = test_model(A, model, criterion, dataset)
        results.append((test_loss, test_acc))

        # log metrics
        tsv_writer.writerow([str(fold),
                             "-1", "-1", "-1", "-1", "-1",
                             str(test_loss), str(test_acc)])

        # reset model and optimizer
        model.reset()
        optimizer.load_state_dict(optimizer_state_zero)

    mean_loss, mean_acc = tuple(sum(e)/len(e) for e in zip(*results))
    tsv_writer.writerow(["-1", "-1", "-1", "-1", "-1", "-1",
                         str(mean_loss), str(mean_acc)])

    return (mean_loss, mean_acc)

def train_model(A, model, optimizer, criterion, dataset, batch_size, nepoch):
    logging.info("Training for {} epoch".format(nepoch))
    # Log wall-clock time
    t0 = time()

    for epoch in range(1, nepoch+1):
        # Single training iteration
        Y_hat = model(dataset['train']['X'], A)

        # Training scores
        train_loss = categorical_crossentropy(Y_hat, dataset['train']['Y'],
                                              dataset['train']['idx'], criterion)
        train_acc = categorical_accuracy(Y_hat, dataset['train']['Y'],
                                         dataset['train']['idx'])

        # validation scores
        val_loss = categorical_crossentropy(Y_hat, dataset['val']['Y'],
                                            dataset['val']['idx'], criterion)
        val_acc = categorical_accuracy(Y_hat, dataset['val']['Y'],
                                         dataset['val']['idx'])

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        logging.info("{:04d} ".format(epoch) \
                     + "| train loss {:.4f} / acc {:.4f} ".format(train_loss,
                                                                  train_acc)
                     + "| val loss {:.4f} / acc {:.4f}".format(val_loss,
                                                               val_acc))

        yield (epoch,
               train_loss, train_acc,
               val_loss, val_acc)

    logging.info("training time: {:.2f}s".format(time()-t0))

def test_model(A, model, criterion, dataset):
    # Predict on full dataset
    Y_hat = model(dataset['test']['X'], A)

    # scores on test set
    test_loss = categorical_crossentropy(Y_hat, dataset['test']['Y'],
                                         dataset['test']['idx'], criterion)
    test_acc = categorical_accuracy(Y_hat, dataset['test']['Y'],
                                    dataset['test']['idx'])

    logging.info("Performance on test set: loss {:.4f} / accuracy {:.4f}".format(
                  test_loss,
                  test_acc))

    return (test_loss, test_acc)

def run(args, tsv_writer, config):
    set_seed(config['task']['seed'])

    featureless = True
    if 'features' in config['graph'].keys() and\
       True in [feature['include'] for feature in config['graph']['features']]:
        featureless = False

    device = torch.device("cpu")
    if config['task']['gpu'] and torch.cuda.is_available():
        device = torch.device("cuda")

    # prep data
    if args.input is None:
        logging.debug("No tarball supplied - building task prequisites")
        with KnowledgeGraph(graph=config['graph']['file']) as kg:
            targets = strip_graph(kg, config)
            A = graph_structure.generate(kg, config)
            X, Y, X_node_map = build_dataset(kg, targets, config, featureless)
    else:
        assert is_readable(args.input)
        logging.debug("Importing prepared tarball")
        with Tarball(args.input, 'r') as tb:
            A = tb.get('A')
            X = tb.get('X')
            Y = tb.get('Y')
            X_node_map = tb.get('X_node_map')

    if featureless:
        # tuple := (shape)
        X = np.empty(X)

    # convert numpy and scipy matrices to pyTorch tensors
    # move to gpu if possible
    A = scipy_sparse_to_pytorch_sparse(A, device)
    X = torch.Tensor(X, device=device) if not featureless else torch.Tensor(X)
    Y = torch.Tensor(Y, device=device)

    if config['task']['kfolds'] < 0:
        loss, accuracy = single_run(A, X, Y, X_node_map, tsv_writer, device,
                                    config, featureless)
    else:
        loss, accuracy = kfold_crossvalidation(A, X, Y, X_node_map,
                                               config['task']['kfolds'],
                                               tsv_writer, device, config,
                                               featureless)

    logging.info("Mean performance: loss {:.4f} / accuracy {:.4f}".format(
                  loss,
                  accuracy))
    if args.verbose < 1:
        print("Mean performance: loss {:.4f} / accuracy {:.4f}".format(
                  loss,
                  accuracy))

def init_logger(filename, verbose=0):
    logging.basicConfig(filename=filename,
                        format='[%(asctime)s] %(module)s/%(funcName)s | %(levelname)s: %(message)s',
                        level=logging.DEBUG)

    if verbose > 0:
        stream_handler = logging.StreamHandler()

        level = logging.INFO
        if verbose >= 2:
            level = logging.DEBUG
        stream_handler.setLevel(level)

        logging.getLogger().addHandler(stream_handler)

if __name__ == "__main__":
    timestamp = int(time())

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Configuration file (toml)", required=True, default=None)
    parser.add_argument("-i", "--input", help="Optional prepared input file (tar)", default=None)
    parser.add_argument("-o", "--output", help="Output directory", default="/tmp/")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action='count', default=0)
    args = parser.parse_args()

    # load configuration
    assert is_readable(args.config)
    config = toml.load(args.config)
    if config['version'] != VERSION:
        raise UserWarning("Supplied config version '{}' differs from expected"+
                          " version '{}'".format(config['version'], VERSION))

    # set output base filename
    baseFilename = "{}{}{}_{}".format(args.output, config['name'], timestamp,\
                                      getpid()) if args.output.endswith("/") \
                    else "{}/{}{}_{}".format(args.output, config['name'],\
                                             timestamp, getpid())
    assert is_writable(baseFilename)

    init_logger(baseFilename+'.log', args.verbose)
    logger = logging.getLogger(__name__)

    tsv_writer = TSV(baseFilename+'.tsv', 'w')

    # log parameters
    logger.debug("Arguments:\n{}".format(
        "\n".join(["\t{}: {}".format(arg, getattr(args, arg)) for arg in vars(args)])))
    logger.debug("Configuration:\n{}".format(
        "\n".join(["\t{}: {}".format(k,v) for k,v in config.items()])))

    # run training
    run(args, tsv_writer, config)

    logging.shutdown()