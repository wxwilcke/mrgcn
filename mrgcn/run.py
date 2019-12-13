#!/usr/bin/env python

import logging
import argparse
from os import getpid
from time import time

import toml
import torch
import torch.nn as nn
import torch.optim as optim

from mrgcn.data.io.knowledge_graph import KnowledgeGraph
from mrgcn.data.io.tarball import Tarball
from mrgcn.data.io.tsv import TSV
from mrgcn.data.utils import is_readable, is_writable, scipy_sparse_to_pytorch_sparse
from mrgcn.encodings import graph_structure
from mrgcn.encodings.graph_features import construct_feature_matrix, features_included
from mrgcn.tasks.config import set_seed
from mrgcn.tasks.node_classification import (build_dataset,
                                             build_model,
                                             categorical_accuracy,
                                             categorical_crossentropy)
from mrgcn.tasks.utils import (mksplits,
                               init_fold,
                               mkfolds,
                               strip_graph,
                               mkbatches,
                               mkbatches_varlength)


VERSION = 0.1

def single_run(A, X, F, Y, C, X_node_map, tsv_writer, device, config,
               modules_config, featureless):
    tsv_writer.writerow(["epoch", "training_loss", "training_accurary",
                                  "validation_loss", "validation_accuracy",
                                  "test_loss", "test_accuracy"])

    # add additional features if available
    F_strings = F.pop("xsd.string", list())
    F_strings = [(f, mkbatches_varlength(*f)) for f in F_strings]
    F_images = F.pop("blob.image", list())
    F_images = [(f, mkbatches(*f)) for f in F_images]
    F_wktLiterals = F.pop("ogc.wktLiteral", list())
    F_wktLiterals = [(f, mkbatches_varlength(*f))
                    for f in F_wktLiterals]

    X = [X, F_strings, F_images, F_wktLiterals]

    # create splits
    dataset = mksplits(X, Y, X_node_map, device,
                            config['task']['dataset_ratio'])

    # compile model
    model = build_model(C, Y, A, modules_config, config, featureless)

    optimizer = optim.Adam(model.parameters(),
                           lr=config['model']['learning_rate'],
                           weight_decay=config['model']['l2norm'])
    criterion = nn.CrossEntropyLoss()

    # train model
    nepoch = config['model']['epoch']

    for epoch in train_model(A, model, optimizer, criterion, dataset,
                             nepoch, device):
        # log metrics
        tsv_writer.writerow([str(epoch[0]),
                             str(epoch[1]),
                             str(epoch[2]),
                             str(epoch[3]),
                             str(epoch[4]),
                             "-1", "-1"])

    # test model
    test_loss, test_acc = test_model(A, model, criterion, dataset, device)
    # log metrics
    tsv_writer.writerow(["-1", "-1", "-1", "-1", "-1",
                         str(test_loss), str(test_acc)])

    return (test_loss, test_acc)

def kfold_crossvalidation(A, X, F, Y, C, X_node_map, k, tsv_writer, device, config,
                          modules_config, featureless):
    tsv_writer.writerow(["fold", "epoch",
                         "training_loss", "training_accurary",
                         "validation_loss", "validation_accuracy",
                         "test_loss", "test_accuracy"])

    # compile model
    model = build_model(C, Y, A, modules_config, config, featureless)

    optimizer = optim.Adam(model.parameters(),
                           lr=config['model']['learning_rate'],
                           weight_decay=config['model']['l2norm'])
    optimizer_state_zero = optimizer.state_dict()  # save initial state
    criterion = nn.CrossEntropyLoss()

    # add additional features if available
    F_strings = F.pop("xsd.string", list())
    F_strings = [(f, mkbatches_varlength(*f)) for f in F_strings]
    F_images = F.pop("blob.image", list())
    F_images = [(f, mkbatches(*f)) for f in F_images]
    F_wktLiterals = F.pop("ogc.wktLiteral", list())
    F_wktLiterals = [(f, mkbatches_varlength(*f))
                    for f in F_wktLiterals]

    X = [X, F_strings, F_images, F_wktLiterals]
    # generate fold indices
    folds_idx = mkfolds(X_node_map.shape[0], k)

    results = []
    logger.info("Starting {}-fold cross validation".format(k))
    for fold in range(1, k+1):
        logger.info("Fold {} / {}".format(fold, k))
        # initialize fold
        dataset = init_fold(X, Y, X_node_map, folds_idx[fold-1],
                            device, config['task']['dataset_ratio'])

        # train model
        nepoch = config['model']['epoch']

        for epoch in train_model(A, model, optimizer, criterion, dataset,
                                 nepoch, device):
            # log metrics
            tsv_writer.writerow([str(fold),
                                 str(epoch[0]),
                                 str(epoch[1]),
                                 str(epoch[2]),
                                 str(epoch[3]),
                                 str(epoch[4]),
                                 "-1", "-1"])

        # test model
        test_loss, test_acc = test_model(A, model, criterion, dataset, device)
        results.append((test_loss, test_acc))

        # log metrics
        tsv_writer.writerow([str(fold),
                             "-1", "-1", "-1", "-1", "-1",
                             str(test_loss), str(test_acc)])

        # reset model and optimizer
        model.init()
        optimizer.load_state_dict(optimizer_state_zero)

    mean_loss, mean_acc = tuple(sum(e)/len(e) for e in zip(*results))
    tsv_writer.writerow(["-1", "-1", "-1", "-1", "-1", "-1",
                         str(mean_loss), str(mean_acc)])

    return (mean_loss, mean_acc)

def train_model(A, model, optimizer, criterion, dataset, nepoch, device):
    logging.info("Training for {} epoch".format(nepoch))
    # Log wall-clock time
    t0 = time()

    model.train(True)
    for epoch in range(1, nepoch+1):
        # Single training iteration
        Y_hat = model(dataset['train']['X'], A, device)

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

    logging.info("training time: {:.2f}s".format(time()-t0))

def test_model(A, model, criterion, dataset, device):
    # Predict on full dataset
    model.train(False)
    with torch.no_grad():
        Y_hat = model(dataset['test']['X'], A, device)

    # scores on test set
    test_loss = categorical_crossentropy(Y_hat, dataset['test']['Y'],
                                         dataset['test']['idx'], criterion)
    test_acc = categorical_accuracy(Y_hat, dataset['test']['Y'],
                                    dataset['test']['idx'])

    test_loss = float(test_loss)
    test_acc = float(test_acc)

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
        logging.debug("Running on GPU")

    # prep data
    if args.input is None:
        logging.debug("No tarball supplied - building task prequisites")
        with KnowledgeGraph(graph=config['graph']['file']) as kg:
            targets = strip_graph(kg, config)
            A, nodes_idx = graph_structure.generate(kg, config)
            F, Y, X_node_map = build_dataset(kg, nodes_idx, targets, config, featureless)
    else:
        assert is_readable(args.input)
        logging.debug("Importing prepared tarball")
        with Tarball(args.input, 'r') as tb:
            A = tb.get('A')
            F = tb.get('F')
            Y = tb.get('Y')
            X_node_map = tb.get('X_node_map')

    # convert numpy and scipy matrices to pyTorch tensors
    num_nodes = Y.shape[0]
    C = 0  # number of columns in X
    X = torch.empty((0,C), dtype=torch.float32)
    modules_config = list()
    if not featureless:
        features_enabled = features_included(config)
        logging.debug("Features included: {}".format(", ".join(features_enabled)))

        X, F = construct_feature_matrix(F, features_enabled, num_nodes)
        X = torch.as_tensor(X)

        # determine configurations
        for datatype in features_enabled:
            if datatype not in F.keys():
                continue

            feature_configs = config['graph']['features']
            feature_config = next((conf for conf in feature_configs
                                   if conf['datatype'] == datatype),
                                  None)
            for encodings, _, c, _ in F[datatype]:
                if datatype in ["xsd.string", "ogc.wktLiteral"]:
                    # stored as list of arrays
                    feature_dim = 0 if datatype == "xsd.string" else 1
                    feature_size = encodings[0].shape[feature_dim]
                    modules_config.append((datatype, (feature_config['batch_size'],
                                                      feature_size,
                                                      c)))
                if datatype in ["blob.image"]:
                    # stored as tensor
                    modules_config.append((datatype, (feature_config['batch_size'],
                                                      encodings.shape[1:],
                                                      c)))

                C += c

    A = scipy_sparse_to_pytorch_sparse(A)
    Y = torch.as_tensor(Y)

    if config['task']['kfolds'] < 0:
        loss, accuracy = single_run(A, X, F, Y, C, X_node_map, tsv_writer, device,
                                    config, modules_config, featureless)
    else:
        loss, accuracy = kfold_crossvalidation(A, X, F, Y, C, X_node_map,
                                               config['task']['kfolds'],
                                               tsv_writer, device, config,
                                               modules_config, featureless)

    if device == torch.device("cuda"):
        logging.debug("Peak GPU memory used (MB): {}".format(
                      str(torch.cuda.max_memory_allocated()/1.0e6)))

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
