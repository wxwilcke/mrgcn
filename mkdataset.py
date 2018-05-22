#!/usr/bin/python3

import logging
import argparse
from time import time

import toml

from data.io.knowledge_graph import KnowledgeGraph
from data.io.tarball import Tarball
from data.utils import is_readable, is_writable
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
    parser.add_argument("-o", "--output", help="Output directory", default="/tmp/")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    args = parser.parse_args()

    assert is_readable(args.config)
    config = toml.load(args.config)

    # set output base filename
    baseFilename = "{}{}{}".format(args.output, config['name'], timestamp) if args.output.endswith("/") \
                    else "{}/{}{}".format(args.output, config['name'], timestamp)
    assert is_writable(baseFilename)

    init_logger(baseFilename+'.log', args.verbose)
    logger = logging.getLogger(__name__)
    

    # log parameters
    logger.info("Arguments:\n{}".format(
        "\n".join(["\t{}: {}".format(arg, getattr(args, arg)) for arg in vars(args)])))
    logger.info("Configuration:\n{}".format(
        "\n".join(["\t{}: {}".format(k,v) for k,v in config.items()])))

    with Tarball(baseFilename+'.tar', 'w') as tb:
        tb.store(run(args, config), names=['A', 'X', 'Y', 'X_node_map'])

    logging.shutdown()
