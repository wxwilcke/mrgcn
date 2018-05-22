#!/usr/bin/python3

import logging
import argparse
from time import time

import toml

from data.io.knowledge_graph import KnowledgeGraph
from data.io.tarball import Tarball
from data.io.tsv import TSV
from data.utils import is_readable, is_writable
from embeddings import graph_structure
from tasks.node_classification import build_dataset, build_model, evaluate_model
from tasks.utils import mksplits, init_fold, mkfolds, sample_mask, set_seed, strip_graph


def single_run(A, X, Y, X_node_map, tsv_writer, config):
    tsv_writer.writerow(["epoch", "training_loss", "training_accurary",
                                  "validation_loss", "validation_accuracy",
                                  "test_loss", "test_accuracy"])

    # create splits
    dataset = mksplits(X, Y, X_node_map, 
                            config['task']['dataset_ratio'])

    # compile model computation graph
    model = build_model(X, Y, A, config)
    
    # train model
    nepoch = config['model']['epoch']
    batch_size = X.shape[0]  # number of nodes
    sample_weights = sample_mask(dataset['train']['X_idx'],
                                            Y.shape[0])

    for epoch in train_model(A, model, dataset, sample_weights, batch_size, nepoch):
        # log metrics
        tsv_writer.writerow([str(epoch[0]), 
                             str(epoch[1]),
                             str(epoch[2]),
                             str(epoch[3]), 
                             str(epoch[4]),
                             "-1", "-1"])

    # test model
    test_loss, test_acc = test_model(A, model, dataset, batch_size)
    # log metrics
    tsv_writer.writerow(["-1", "-1", "-1", "-1", "-1", 
                         str(test_loss[0]), str(test_acc[0])])

    return (test_loss[0], test_acc[0])

def kfold_crossvalidation(A, X, Y, X_node_map, k, tsv_writer, config):
    tsv_writer.writerow(["fold", "epoch", 
                         "training_loss", "training_accurary",
                         "validation_loss", "validation_accuracy",
                         "test_loss", "test_accuracy"])

    # generate fold indices
    folds_idx = mkfolds(X_node_map.shape[0], k)

    results = []
    logger.info("Starting {}-fold cross validation".format(k))
    for fold in range(1, k+1):
        logger.info("Fold {} / {}".format(fold, k))

        # compile model computation graph
        model = build_model(X, Y, A, config)
        
        # initialize fold 
        dataset = init_fold(X, Y, X_node_map, folds_idx[fold-1],
                            config['task']['dataset_ratio'])

        # train model
        nepoch = config['model']['epoch']
        batch_size = X.shape[0]  # number of nodes
        sample_weights = sample_mask(dataset['train']['X_idx'],
                                                Y.shape[0])

        for epoch in train_model(A, model, dataset, sample_weights, batch_size, nepoch):
            # log metrics
            tsv_writer.writerow([str(fold),
                                 str(epoch[0]), 
                                 str(epoch[1]),
                                 str(epoch[2]),
                                 str(epoch[3]), 
                                 str(epoch[4]),
                                 "-1", "-1"])

        # test model
        test_loss, test_acc = test_model(A, model, dataset, batch_size)
        results.append((test_loss[0], test_acc[0]))

        # log metrics
        tsv_writer.writerow([str(fold), 
                             "-1", "-1", "-1", "-1", "-1", 
                             str(test_loss[0]), str(test_acc[0])])

    mean_loss, mean_acc = tuple(sum(e)/len(e) for e in zip(*results))
    tsv_writer.writerow(["-1", "-1", "-1", "-1", "-1", "-1", 
                         str(mean_loss), str(mean_acc)])

    return (mean_loss, mean_acc)

def train_model(A, model, dataset, sample_weights, batch_size, nepoch):
    logging.info("Training for {} epoch".format(nepoch))
    # Log wall-clock time
    t0 = time()
    for epoch in range(1, nepoch+1):
        # Single training iteration
        model.fit(x=[dataset['train']['X']] + A, 
                  y=dataset['train']['Y'],
                  batch_size=batch_size,
                  epochs=1,
                  shuffle=False,
                  sample_weight=sample_weights,
                  validation_data=([dataset['val']['X']] + A, 
                                    dataset['val']['Y']),
                  callbacks=[],
                  verbose=0)

        # Predict on full dataset
        Y_hat = model.predict(x=[dataset['train']['X']] + A, 
                              batch_size=batch_size,
                              verbose=0)

        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_model(Y_hat, 
                                                       [dataset['train']['Y'],
                                                        dataset['val']['Y']],
                                                       [dataset['train']['X_idx'],
                                                        dataset['val']['X_idx']])
    
        logging.info("{:04d} ".format(epoch) \
                     + "| train loss {:.4f} / acc {:.4f} ".format(train_val_loss[0],
                                                                  train_val_acc[0])
                     + "| val loss {:.4f} / acc {:.4f}".format(train_val_loss[1],
                                                               train_val_acc[1]))

        yield (epoch,
               train_val_loss[0], train_val_acc[0], 
               train_val_loss[1], train_val_acc[1])
    
    logging.info("training time: {:.2f}s".format(time()-t0))

def test_model(A, model, dataset, batch_size):
    # Predict on full dataset
    Y_hat = model.predict(x=[dataset['train']['X']] + A, 
                          batch_size=batch_size,
                          verbose=0)

    test_loss, test_acc = evaluate_model(Y_hat, 
                                         [dataset['test']['Y']],
                                         [dataset['test']['X_idx']])
    
    logging.info("Performance on test set: loss {:.4f} / accuracy {:.4f}".format(
                  test_loss[0],
                  test_acc[0]))

    return (test_loss, test_acc)


def run(args, tsv_writer, config):
    set_seed(config['task']['seed'])

    # prep data
    if args.input is None:
        logging.debug("No tarball supplied - building task prequisites")
        with KnowledgeGraph(path=config['graph']['file']) as kg:
            targets = strip_graph(kg, config)
            A = graph_structure.generate(kg, config)
            X, Y, X_node_map = build_dataset(kg, targets, config)
    else:
        assert is_readable(args.input)
        logging.debug("Importing prepared tarball")
        with Tarball(args.input, 'r') as tb:
            A = tb.get('A')
            X = tb.get('X')
            Y = tb.get('Y')
            X_node_map = tb.get('X_node_map')
    
    if config['task']['kfolds'] < 0:
        loss, accuracy = single_run(A, X, Y, X_node_map, tsv_writer, config)
    else:
        loss, accuracy = kfold_crossvalidation(A, X, Y, X_node_map,
                                               config['task']['kfolds'],
                                               tsv_writer, config)

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

    # set output base filename
    baseFilename = "{}{}{}".format(args.output, config['name'], timestamp) if args.output.endswith("/") \
                    else "{}/{}{}".format(args.output, config['name'], timestamp)
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
