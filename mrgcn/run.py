#!/usr/bin/env python

import logging
import argparse
from os import getpid
from time import time

import toml
import torch

from mrgcn.data.io.tarball import Tarball
from mrgcn.data.io.tsv import TSV
from mrgcn.data.utils import (is_readable,
                              is_writable,
                              scipy_sparse_to_pytorch_sparse,
                              set_seed,
                              setup_features)
import mrgcn.tasks.node_classification as node_classification
import mrgcn.tasks.link_prediction as link_prediction

def run(A, X, Y, C, data, tsv_writer, device, config,
        modules_config, featureless, test_split):
    tsv_writer.writerow(["epoch", "training_loss", "training_accurary",
                                  "validation_loss", "validation_accuracy",
                                  "test_loss", "test_accuracy"])

    task = config['task']['type']
    logger.info("Starting {} task".format(task))
    if task == "node classification":
        loss, acc = node_classification.run(A, X, Y, C, tsv_writer,
                                              device, config,
                                              modules_config,
                                              featureless, test_split)
        return (loss, acc)

    elif task == "link prediction":
        mrr, hits_at_k = link_prediction.run(A, X, C, data,
                                               tsv_writer, device, config,
                                               modules_config,
                                               featureless, test_split)
        return (mrr, hits_at_k)

def main(args, tsv_writer, config):
    set_seed(config['task']['seed'])

    test_split = 'test' if args.test else 'valid' # use test split?

    featureless = True
    if 'features' in config['graph'].keys() and\
       True in [feature['include'] for feature in config['graph']['features']]:
        featureless = False

    device = torch.device("cpu")
    if config['task']['gpu'] and torch.cuda.is_available():
        device = torch.device("cuda")
        logging.debug("Running on GPU")

    assert is_readable(args.input)
    logging.debug("Importing tarball")
    with Tarball(args.input, 'r') as tb:
        A = tb.get('A')
        F = tb.get('F')
        Y = tb.get('Y')  # empty if doing link prediction
        data = tb.get('data')  # empty if doing node classification

    # prep data
    num_nodes = A.shape[0]
    A = scipy_sparse_to_pytorch_sparse(A)
    X, C, modules_config = setup_features(F, num_nodes, featureless, config)
    if len(X) <= 1 and X[0].size(1) <= 0:
        featureless = True

    task = config['task']['type']
    out = run(A, X, Y, C, data, tsv_writer, device,
              config, modules_config, featureless, test_split)

    if task == "node classification":
        print("loss {:.4f} / accuracy {:.4f}".format(out[0], out[1]))
    elif task == "link prediction":
        print("MRR (raw) {:.4f}".format(out[0])
              + " / " + " / ".join(["H@{} {:.4f}".format(k,v) for
                                          k,v in out[1].items()]))

def init_logger(filename, dry_run, verbose=0):
    handlers = list()
    if verbose > 0:
        stream_handler = logging.StreamHandler()

        level = logging.INFO
        if verbose >= 2:
            level = logging.DEBUG
        stream_handler.setLevel(level)

        handlers.append(stream_handler)

    if not dry_run:
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s] %(module)s/%(funcName)s | %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)

        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=handlers)

if __name__ == "__main__":
    timestamp = int(time())

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Configuration file (toml)", required=True, default=None)
    parser.add_argument("-i", "--input", help="Optional prepared input file (tar)", default=None)
    parser.add_argument("-o", "--output", help="Output directory", default="/tmp/")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action='count', default=0)
    parser.add_argument("--dry_run", help="Suppress writing output files to disk",
                        action='store_true')
    parser.add_argument("--test", help="Report accuracy on test set rather than on validation set",
                        action='store_true')
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

    init_logger(baseFilename+'.log', args.dry_run, args.verbose)
    logger = logging.getLogger()

    tsv_writer = TSV(baseFilename+'.tsv', 'w', args.dry_run)

    # log parameters
    logger.debug("Arguments:\n{}".format(
        "\n".join(["\t{}: {}".format(arg, getattr(args, arg)) for arg in vars(args)])))
    logger.debug("Configuration:\n{}".format(
        "\n".join(["\t{}: {}".format(k,v) for k,v in config.items()])))

    # run training
    main(args, tsv_writer, config)

    logging.shutdown()
