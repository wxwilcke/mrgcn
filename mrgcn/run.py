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
from mrgcn.data.utils import (is_readable,
                              is_writable,
                              scipy_sparse_to_pytorch_sparse,
                              merge_sparse_encodings_sets,
                              set_seed)
from mrgcn.encodings import graph_structure
from mrgcn.encodings.graph_features import construct_feature_matrix, features_included
from mrgcn.tasks.node_classification import (build_dataset,
                                             build_model,
                                             categorical_accuracy,
                                             categorical_crossentropy)
from mrgcn.tasks.utils import (strip_graph,
                               mkbatches,
                               mkbatches_varlength,
                               remove_outliers)


def single_run(A, X, Y, C, tsv_writer, device, config,
               modules_config, featureless):
    tsv_writer.writerow(["epoch", "training_loss", "training_accurary",
                                  "validation_loss", "validation_accuracy",
                                  "test_loss", "test_accuracy"])

    # compile model
    model = build_model(C, Y, A, modules_config, config, featureless)
    modules = {(name.split('.')[1], module)
               for name, module in model.named_modules() if len(name.split('.')) == 2}
    params = list()
    for name, module in modules:
        if name.startswith("CNN1D"):
            params.append({"params": module.parameters(), "lr": 1e-3})
            continue
        if name.startswith("RNN"):
            params.append({"params": module.parameters(), "lr": 1e-3})
            continue
        params.append({"params": module.parameters()})
    optimizer = optim.Adam(params,
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
    model.train(True)
    for epoch in range(1, nepoch+1):
        batch_grad_idx = epoch - 1
        if not mini_batch:
            batch_grad_idx = -1

        # Single training iteration
        Y_hat = model(X, A,
                      batch_grad_idx=batch_grad_idx,
                      device=device)

        # Training scores
        train_loss = categorical_crossentropy(Y_hat, Y['train'], criterion)
        train_acc = categorical_accuracy(Y_hat, Y['train'])

        # validation scores
        val_loss = categorical_crossentropy(Y_hat, Y['valid'], criterion)
        val_acc = categorical_accuracy(Y_hat, Y['valid'])

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

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
        target_triples = dict()
        for split in ("train", "valid", "test"):
            with KnowledgeGraph(graph=config['graph'][split]) as kg_split:
                target_triples[split] = frozenset(kg_split.graph)

        with KnowledgeGraph(graph=config['graph']['file']) as kg:
            strip_graph(kg, config)
            A, nodes_idx = graph_structure.generate(kg, config)
            F, Y = build_dataset(kg, nodes_idx, target_triples, config, featureless)
    else:
        assert is_readable(args.input)
        logging.debug("Importing prepared tarball")
        with Tarball(args.input, 'r') as tb:
            A = tb.get('A')
            F = tb.get('F')
            Y = tb.get('Y')

    # convert numpy and scipy matrices to pyTorch tensors
    num_nodes = Y["train"].shape[0]
    C = 0  # number of columns in X
    X = [torch.empty((num_nodes,C), dtype=torch.float32)]
    modules_config = list()
    if not featureless:
        features_enabled = features_included(config)
        logging.debug("Features included: {}".format(", ".join(features_enabled)))

        X, F = construct_feature_matrix(F, features_enabled, num_nodes,
                                        config['graph']['features'])
        C += X.shape[1]
        X = [torch.as_tensor(X)]

        # determine configurations
        for datatype in features_enabled:
            if datatype not in F.keys():
                continue

            logger.debug("Found {} encoding set(s) for datatype {}".format(
                len(F[datatype]),
                datatype))

            if datatype not in ['xsd.string', 'ogc.wktLiteral', 'blob.image']:
                continue

            feature_configs = config['graph']['features']
            feature_config = next((conf for conf in feature_configs
                                   if conf['datatype'] == datatype),
                                  None)

            # preprocess
            encoding_sets = F.pop(datatype, list())
            if feature_config['share_weights'] and datatype == "xsd.string":
                # note: images and geometries always share weights atm
                logger.debug("weight sharing enabled for {}".format(datatype))
                encoding_sets = merge_sparse_encodings_sets(encoding_sets)

            for encodings, _, c, _, _ in encoding_sets:
                if datatype in ["xsd.string", "ogc.wktLiteral"]:
                    # stored as list of arrays
                    feature_dim = 0 if datatype == "xsd.string" else 1
                    feature_size = encodings[0].shape[feature_dim]
                    modules_config.append((datatype, (feature_config['passes_per_batch'],
                                                      feature_size,
                                                      c)))
                if datatype in ["blob.image"]:
                    # stored as tensor
                    modules_config.append((datatype, (feature_config['passes_per_batch'],
                                                      encodings.shape[1:],
                                                      c)))

                C += c

            # remove outliers?
            if feature_config['remove_outliers']:
                encoding_sets = [remove_outliers(*f) for f in encoding_sets]

            nepoch = config['model']['epoch']
            encoding_sets = [(f, mkbatches(*f,
                                           nepoch=nepoch,
                                           passes_per_batch=feature_config['passes_per_batch']))
                             for f in encoding_sets] if datatype == "blob.image"\
                    else [(f, mkbatches_varlength(*f,
                                                  nepoch=nepoch,
                                                  passes_per_batch=feature_config['passes_per_batch']))
                          for f in encoding_sets]

            X.append((datatype, encoding_sets))

    if len(X) <= 1 and X[0].size(1) <= 0:
        featureless = True

    A = scipy_sparse_to_pytorch_sparse(A)

    loss, accuracy = single_run(A, X, Y, C, tsv_writer, device,
                                config, modules_config, featureless)

    if device == torch.device("cuda"):
        logging.debug("Peak GPU memory used (MB): {}".format(
                      str(torch.cuda.max_memory_allocated()/1.0e6)))

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
