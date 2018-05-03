#!/usr/bin/python3

import logging
import argparse
from os.path import splitext
from time import time

from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
import toml

from data.io.knowledge_graph import KnowledgeGraph
from data.io.tarball import Tarball
from data.utils import is_readable, is_writable
from embeddings import graph_structure
from tasks.node_classification import build_dataset, build_model
from tasks.utils import create_splits, sample_mask, set_seed, strip_graph

def run(args, csv_log, config):
    set_seed(config['task']['seed'])

    # prep data
    if args.input is None:
        logging.info("No tarball supplied - building task prequisites")
        with KnowledgeGraph(path=config['graph']['file']) as kg:
            targets = strip_graph(kg, config)
            A = graph_structure.generate(kg, config)
            X, Y, X_node_map = build_dataset(kg, targets, config)
    else:
        assert is_readable(args.input)
        logging.info("Importing prepared tarball")
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

    adapt_lr = ReduceLROnPlateau(monitor='val_loss',
                                 factor=0.4,
                                 patience=5,
                                 min_lr=0.001,
                                 verbose=int(args.verbose))
    early_stop = EarlyStopping(monitor='val_loss', 
                               min_delta=0.0001, 
                               patience=10, 
                               verbose=int(args.verbose))

    logging.info("Training for a maximum of {} epoch".format(nepochs))
    training = model.fit(x=[dataset['train']['X']] + A, 
                  y=dataset['train']['Y'],
                  batch_size=batch_size,
                  nb_epoch=nepochs,
                  sample_weight=sample_mask(dataset['train']['X_idx'],
                                            Y.shape[0]),
                  validation_data=([dataset['val']['X']] + A, 
                                   dataset['val']['Y']),
                  callbacks=[adapt_lr, early_stop, csv_log],
                  verbose=int(args.verbose))

    # test model
    testing = model.evaluate(x=[dataset['test']['X']] + A, 
                  y=dataset['test']['Y'],
                  batch_size=batch_size,
                  verbose=int(args.verbose))

    results = "Performance on test set:\n{}".format(
        "\n".join(["\t{}: {:.4f}".format(k,v) for k,v in zip(
                                                         model.metrics_names,
                                                         testing)]))

    logger.info(results)
    print(results)
    
    return model

def set_logging(args, timestamp):
    log_path = args.log_directory
    if not is_writable(log_path):
        return

    filename = "{}{}.log".format(log_path, timestamp) if log_path.endswith("/") \
                    else "{}/{}.log".format(log_path, timestamp)

    logging.basicConfig(filename=filename,
                        format='%(asctime)s %(levelname)s [%(module)s/%(funcName)s]: %(message)s',
                        level=logging.INFO)

    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler())

if __name__ == "__main__":
    timestamp = int(time())

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Configuration file (toml)", required=True, default=None)
    parser.add_argument("-i", "--input", help="Prepared input file (tar)", default=None)
    parser.add_argument("-m", "--mode", help="Train or test a model", choices=("train", "test"), default="train")
    parser.add_argument("-o", "--output", help="Output file (csv)", default=None)
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    parser.add_argument("--log_directory", help="Where to save the log file", default="../log/")
    parser.add_argument("--save_model", help="Save the trained model to file (h5)", action="store_true")
    args = parser.parse_args()

    assert is_readable(args.config)
    config = toml.load(args.config)

    if args.output is None:
        args.output = './' + config['name'] + '{}.tsv'.format(timestamp)
    assert is_writable(args.output)
    csv_log = CSVLogger(args.output, separator=',')

    set_logging(args, timestamp)
    logger = logging.getLogger(__name__)
    logger.info("Arguments:\n{}".format(
        "\n".join(["\t{}: {}".format(arg, getattr(args, arg)) for arg in vars(args)])))
    logger.info("Configuration:\n{}".format(
        "\n".join(["\t{}: {}".format(k,v) for k,v in config.items()])))

    # write results
    model = run(args, csv_log, config)
    
    if args.save_model is True:
        raise NotImplementedError()
        output = splitext(args.output)[1] + '.h5'
        assert is_writable(output)
        
        logger.info("Saving trained model to {}".format(output))
        model.save_model(output)
    
    logging.shutdown()
