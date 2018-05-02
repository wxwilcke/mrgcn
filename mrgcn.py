#!/usr/bin/python3

import logging
import argparse
from os.path import splitext
from time import time

import toml

from data.readers.knowledge_graph import KnowledgeGraph
from data.utils import is_readable, is_writable
from data.readers import tar
#from data.writers import tsv
from embeddings import graph_structure
from tasks.node_classification import build_dataset, build_model
from tasks.utils import create_splits, sample_mask, set_seed, strip_graph

def run(args, config):
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
        tarball = tar.read(args.input)
        A = tarball['A']
        X = tarball['X']
        Y = tarball['Y']
        X_node_map = tarball['X_node_map']

    # compile model computation graph
    model = build_model(X, Y, A, config)

    # create splits
    dataset = create_splits(X, Y, X_node_map, config['task']['dataset_ratio'])

    # train model
    nepochs = config['model']['epoch']
    batch_size = X.shape[0]  # number of nodes

    logging.info("Training for {} epoch".format(nepochs))
    training = model.fit(x=[dataset['train']['X']] + A, 
                  y=dataset['train']['Y'],
                  batch_size=batch_size,
                  nb_epoch=nepochs,
                  sample_weight=sample_mask(dataset['train']['X_idx'],
                                            Y.shape[0]),
                  validation_data=([dataset['val']['X']] + A, 
                                   dataset['val']['Y']),
                  verbose=int(args.verbose))

    print(training.history)

    testing = model.evaluate(x=[dataset['test']['X']] + A, 
                  y=dataset['test']['Y'],
                  batch_size=batch_size,
                  verbose=int(args.verbose))

    print(model.metrics_names)
    print(testing)

    return (model, testing)

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
    parser.add_argument("-o", "--output", help="Output file (tsv)", default=None)
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    parser.add_argument("--log_directory", help="Where to save the log file", default="../log/")
    parser.add_argument("--save_model", help="Save the trained model to file (h5)", action="store_true")
    args = parser.parse_args()

    assert is_readable(args.config)
    config = toml.load(args.config)

    if args.output is None:
        args.output = './' + config['name'] + '{}.tsv'.format(timestamp)
    assert is_writable(args.output)

    set_logging(args, timestamp)
    logger = logging.getLogger(__name__)
    logger.info("Arguments:\n{}".format(
        "\n".join(["\t{}: {}".format(arg, getattr(args, arg)) for arg in vars(args)])))
    logger.info("Configuration:\n{}".format(
        "\n".join(["\t{}: {}".format(k,v) for k,v in config.items()])))

    # write results
    model, results = run(args, config)
    #tsv.write(args.output, results)
    
    if args.save_model is True:
        raise NotImplementedError()
        output = splitext(args.output)[1] + '.h5'
        assert is_writable(output)
        
        logger.info("Saving trained model to {}".format(output))
        model.save_model(output)
    
    logging.shutdown()
