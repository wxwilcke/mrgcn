#!/usr/bin/python3

import logging
import argparse
from os.path import splitext
from time import time

import toml

from data.io.knowledge_graph import KnowledgeGraph
from data.io.tarball import Tarball
from data.io.tsv import TSV
from data.utils import is_readable, is_writable
from embeddings import graph_structure
from tasks.node_classification import build_dataset, build_model, evaluate_model
from tasks.utils import create_splits, sample_mask, set_seed, strip_graph

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

    # compile model computation graph
    model = build_model(X, Y, A, config)

    # create splits
    dataset = create_splits(X, Y, X_node_map, config['task']['dataset_ratio'])

    # train model
    nepochs = config['model']['epoch']
    batch_size = X.shape[0]  # number of nodes
    sample_weights = sample_mask(dataset['train']['X_idx'],
                                            Y.shape[0])

    logging.info("Training for {} epoch".format(nepochs))
    tsv_writer.writerow(["epoch", "training_loss", "training_accurary",
                                  "validation_loss", "validation_accuracy",
                                  "test_loss", "test_accuracy"])
    # Log wall-clock time
    t0 = time()
    for epoch in range(1, nepochs+1):
        # Single training iteration
        model.fit(x=[dataset['train']['X']] + A, 
                  y=dataset['train']['Y'],
                  batch_size=batch_size,
                  nb_epoch=1,
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

        # log metrics
        tsv_writer.writerow([str(epoch), 
                             str(train_val_loss[0]),
                             str(train_val_acc[0]),
                             str(train_val_loss[1]), 
                             str(train_val_acc[1]),
                             "-1", "-1"])

        logging.info("{:04d} ".format(epoch) \
                     + "| train loss {:.4f} / acc {:.4f} ".format(train_val_loss[0],
                                                                  train_val_acc[0])
                     + "| val loss {:.4f} / acc {:.4f}".format(train_val_loss[1],
                                                               train_val_acc[1]))

    logging.info("training time (s): {:.2f}".format(time()-t0))

    # Testing
    test_loss, test_acc = evaluate_model(Y_hat, 
                                         [dataset['test']['Y']],
                                         [dataset['test']['X_idx']])

    # log metrics
    tsv_writer.writerow(["-1", "-1", "-1", "-1", "-1", 
                         str(test_loss[0]), str(test_acc[0])])

    logging.info("Performance on test set: loss {:.4f} / accuracy {:.4f}".format(
                  test_loss[0],
                  test_acc[0]))
    if args.verbose < 1:
        print("Performance on test set: loss {:.4f} / accuracy {:.4f}".format(
                  test_loss[0],
                  test_acc[0]))

    return model

def set_logging(args, timestamp):
    log_path = args.log_directory
    if not is_writable(log_path):
        return

    filename = "{}{}.log".format(log_path, timestamp) if log_path.endswith("/") \
                    else "{}/{}.log".format(log_path, timestamp)
    
    logging.basicConfig(filename=filename,
                        format='[%(asctime)s] %(module)s/%(funcName)s | %(levelname)s: %(message)s',
                        level=logging.DEBUG)

    if args.verbose > 0:
        stream_handler = logging.StreamHandler()

        level = logging.INFO
        if args.verbose >= 2:
            level = logging.DEBUG
        stream_handler.setLevel(level)
        
        logging.getLogger().addHandler(stream_handler)

if __name__ == "__main__":
    timestamp = int(time())

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Configuration file (toml)", required=True, default=None)
    parser.add_argument("-i", "--input", help="Prepared input file (tar)", default=None)
    parser.add_argument("-m", "--mode", help="Train or test a model", choices=("train", "test"), default="train")
    parser.add_argument("-o", "--output", help="Output file (tsv)", default=None)
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action='count', default=0)
    parser.add_argument("--log_directory", help="Where to save the log file", default="../log/")
    parser.add_argument("--save_model", help="Save the trained model to file (h5)", action="store_true")
    args = parser.parse_args()

    assert is_readable(args.config)
    config = toml.load(args.config)

    if args.output is None:
        args.output = './' + config['name'] + '{}.tsv'.format(timestamp)
    assert is_writable(args.output)
    tsv_writer = TSV(args.output, 'w')

    set_logging(args, timestamp)
    logger = logging.getLogger(__name__)
    logger.debug("Arguments:\n{}".format(
        "\n".join(["\t{}: {}".format(arg, getattr(args, arg)) for arg in vars(args)])))
    logger.debug("Configuration:\n{}".format(
        "\n".join(["\t{}: {}".format(k,v) for k,v in config.items()])))

    # write results
    model = run(args, tsv_writer, config)
    
    if args.save_model is True:
        raise NotImplementedError()
        output = splitext(args.output)[1] + '.h5'
        assert is_writable(output)
        
        logger.info("Saving trained model to {}".format(output))
        model.save_model(output)
    
    logging.shutdown()
