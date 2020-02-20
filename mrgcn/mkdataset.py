#!/usr/bin/python3

import logging
import argparse
from time import time

import toml

from mrgcn.data.io.knowledge_graph import KnowledgeGraph
from mrgcn.data.io.tarball import Tarball
from mrgcn.data.utils import is_readable, is_writable
from mrgcn.encodings import graph_structure
import mrgcn.tasks.node_classification as node_classification
import mrgcn.tasks.link_prediction as link_prediction
from mrgcn.tasks.utils import strip_graph, triples_to_indices

def run(args, config):
    task = config['task']['type']
    logger.info("Task set to {}".format(task))
    logger.info("Generating data structures")

    featureless = True
    if 'features' in config['graph'].keys() and\
       True in [feature['include'] for feature in config['graph']['features']]:
        featureless = False

    data = None
    if task == "node classification":
        triples = dict()
        for split in ("train", "valid", "test"):
            with KnowledgeGraph(graph=config['graph'][split]) as kg_split:
                triples[split] = frozenset(kg_split.graph)

        with KnowledgeGraph(graph=config['graph']['context']) as kg:
            strip_graph(kg, config)
            A, nodes_map, _ = graph_structure.generate(kg, config)
            F, Y = node_classification.build_dataset(kg, nodes_map, triples, config,
                                                     featureless)
    elif task == "link prediction":
        with KnowledgeGraph([config['graph']['train'],
                             config['graph']['valid'],
                             config['graph']['test']]) as kg:
            A, nodes_map, edges_map = graph_structure.generate(kg, config)
            F, Y = link_prediction.build_dataset(kg, nodes_map,
                                                 config, featureless)

        separate_literals = config['graph']['structural']['separate_literals']
        data = dict()
        for split in ("train", "valid", "test"):
            with KnowledgeGraph(graph=config['graph'][split]) as kg_split:
                data[split] = triples_to_indices(kg_split, nodes_map, edges_map,
                                                 separate_literals)

    return (A, F, Y, data)

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
        tb.store(run(args, config), names=['A', 'F', 'Y', 'data'])

    logging.shutdown()
