#!/usr/bin/python3

import logging
import argparse
from time import time

import toml

from data.readers.knowledge_graph import KnowledgeGraph
from data.utils import is_readable, is_writable
from data.writers import tar
from embeddings import graph_structure
from tasks.node_classification import build_dataset
from tasks.utils import strip_graph

def run(args, config):
    logger.info("Generating data structures")
    with KnowledgeGraph(path=config['graph']['file']) as kg:
        targets = strip_graph(kg, config)
        A = graph_structure.generate(kg, config)
        X, Y, X_node_map = build_dataset(kg, targets, config)

    return (A, X, Y, X_node_map)

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
    parser.add_argument("-c", "--config", help="Configuration file (toml)", default=None)
    parser.add_argument("-o", "--output", help="Output file (tar)", default=None)
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    parser.add_argument("--log_directory", help="Where to save the log file", default="../log/")
    args = parser.parse_args()

    assert is_readable(args.config)
    config = toml.load(args.config)

    if args.output is None:
        args.output = './' + config['name'] + '{}.tar'.format(timestamp)
    assert is_writable(args.output)

    set_logging(args, timestamp)
    logger = logging.getLogger(__name__)
    logger.info("Arguments:\n{}".format(
        "\n".join(["\t{}: {}".format(arg, getattr(args, arg)) for arg in vars(args)])))
    logger.info("Configuration:\n{}".format(
        "\n".join(["\t{}: {}".format(k,v) for k,v in config.items()])))

    tar.store(args.output, run(args, config), names=['A', 'X', 'Y', 'X_node_map'])

    logging.shutdown()
