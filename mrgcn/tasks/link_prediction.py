#!/usr/bin/python3

from copy import deepcopy
import logging
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mrgcn.encodings.graph_features import construct_features
from mrgcn.models.mrgcn import MRGCN


logger = logging.getLogger(__name__)

def run(A, X, C, data, tsv_writer, device, config,
        modules_config, featureless, test_split):
    tsv_writer.writerow(["epoch", "train_loss",
                         "valid_mrr_raw", "valid_H@1", "valid_H@3", "valid_H@10",
                         "test_mrr_raw", "test_H@1", "test_H@3", "test_H@10"])

    # compile model
    num_nodes = A.shape[0]
    model = build_model(C, A, modules_config, config, featureless)
    optimizer = optim.Adam(model.parameters(),
                           lr=config['model']['learning_rate'],
                           weight_decay=config['model']['l2norm'])
    criterion = nn.BCEWithLogitsLoss()

    # mini batching
    distmult_batch_size = config['model']['distmult_batch_size']
    mrr_batch_size = config['model']['mrr_batch_size']

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
    for epoch in train_model(A, X, data, num_nodes, model, optimizer,
                             criterion, nepoch,
                             distmult_batch_size, mrr_batch_size, device):
        # log metrics
        tsv_writer.writerow([str(epoch[0]),
                             str(epoch[1]),
                             str(epoch[2]),
                             str(epoch[3]),
                             str(epoch[4]),
                             str(epoch[5]),
                             "-1", "-1", "-1", "-1"])

        # early stopping
        val_mrr = epoch[2]
        if patience <= 0:
            continue
        if best_score < 0:
            best_score = val_mrr
            best_state = model.state_dict()
        if val_mrr <= best_score + delta:
            patience_left -= 1
        else:
            best_score = val_mrr
            best_state = model.state_dict()
            patience_left = patience
        if patience_left <= 0:
            model.load_state_dict(best_state)
            logger.info("Early stopping after no improvement for {} epoch".format(patience))
            break

    logging.info("Training time: {:.2f}s".format(time()-t0))

    # test model
    mrr, hits_at_k = test_model(A, X, data, num_nodes, model,
                                criterion, mrr_batch_size, test_split, device)
    # log metrics
    tsv_writer.writerow(["-1", "-1", "-1", "-1", "-1", "-1",
                         str(mrr), str(hits_at_k[1]),
                         str(hits_at_k[3]), str(hits_at_k[10])])

    return (mrr, hits_at_k)

def train_model(A, X, data, num_nodes, model, optimizer, criterion,
                nepoch, distmult_batch_size, mrr_batch_size,
                device):
    logging.info("Training for {} epoch".format(nepoch))

    # create batches
    batches = dict()
    for split, batch_size in zip(('train', 'valid'),
                                 (distmult_batch_size, mrr_batch_size)):
        nsamples = data[split].shape[0]
        nbatches = max(1, nsamples//batch_size)
        batches[split] = np.array_split(np.arange(nsamples),
                                        nbatches)

    nsamples_train = data['train'].shape[0]
    nbatches_train = len(batches["train"])
    nbatches_valid = len(batches["valid"])
    for epoch in range(1, nepoch+1):
        model.train()
        # Single iteration
        node_embeddings = model(X, A, device=device)
        edge_embeddings = model.rgcn.relations
        train_data = deepcopy(data["train"])

        # sample negative triples
        ncorrupt = nsamples_train//5
        neg_samples_idx = np.random.choice(np.arange(nsamples_train),
                                           ncorrupt,
                                           replace=False)

        mask = np.random.choice(np.array([0, 1], dtype=np.bool),
                                ncorrupt)
        corrupt_head_idx = neg_samples_idx[mask]
        corrupt_tail_idx = neg_samples_idx[~mask]

        train_data[corrupt_head_idx, 0] = np.random.choice(np.arange(num_nodes),
                                                           len(corrupt_head_idx))
        train_data[corrupt_tail_idx, 2] = np.random.choice(np.arange(num_nodes),
                                                           len(corrupt_tail_idx))

        # create labels; positive samples are 1, negative 0
        Y = torch.ones(nsamples_train, dtype=torch.float32)
        Y[neg_samples_idx] = 0

        # compute DistMult score for batch
        Y_hat = torch.empty(nsamples_train, dtype=torch.float32)
        for train_batch_id, train_batch in enumerate(batches["train"], 1):
            logger.debug(" DistMult train batch {} / {}".format(train_batch_id,
                                                                nbatches_train))

            batch_data = train_data[train_batch]

            # compute score
            Y_hat[train_batch] = score_distmult(node_embeddings[batch_data[:, 0]],
                                                edge_embeddings[batch_data[:, 1]],
                                                node_embeddings[batch_data[:, 2]])

        # compute loss
        loss = binary_crossentropy(Y_hat, Y, criterion)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()  # training loss
        optimizer.step()

        loss = float(loss)  # remove pointer to gradients to free memory

        # validate
        model.eval()
        with torch.no_grad():
            node_embeddings = model(X, A, device=device)
            edge_embeddings = model.rgcn.relations
            valid_data = deepcopy(data["valid"])

            ranks = list()
            for valid_batch_id, valid_batch in enumerate(batches["valid"], 1):
                logger.debug(" DistMult valid batch {} / {}".format(valid_batch_id,
                                                                    nbatches_valid))
                batch_data = valid_data[valid_batch]

                h = batch_data[:, 0]
                r = batch_data[:, 1]
                t = batch_data[:, 2]

                ranks_hr = compute_ranks(h, r, t, node_embeddings, edge_embeddings)
                ranks_rt = compute_ranks(t, r, h, node_embeddings, edge_embeddings)
                ranks.append(torch.cat([ranks_hr, ranks_rt]))

            ranks = torch.cat(ranks)
            mrr_raw = torch.mean(1.0 / ranks.float())

            hits_at_k = dict()
            for k in [1, 3, 10]:
                hits_at_k[k] = float(torch.mean((ranks <= k).float()))

        logging.info("{:04d} ".format(epoch) \
                     + "| train loss {:.4f} ".format(loss)
                     + "| valid MRR (raw) {:.4f} ".format(mrr_raw)
                     + "/ " + " / ".join(["H@{} {:.4f}".format(k,v)
                                         for k,v in hits_at_k.items()]))

        yield (epoch, loss, mrr_raw, hits_at_k[1], hits_at_k[3], hits_at_k[10])

def test_model(A, X, data, num_nodes, model, criterion,
               mrr_batch_size, test_split, device):
    nsamples = data[test_split].shape[0]
    batches = np.array_split(np.arange(nsamples),
                             max(1, nsamples//mrr_batch_size))
    nbatches = len(batches)

    model.eval()
    mrr = 0.0
    hits_at_k = {1: 0.0, 3: 0.0, 10: 0.0}
    with torch.no_grad():
        node_embeddings = model(X, A, device=device)
        edge_embeddings = model.rgcn.relations
        test_data = deepcopy(data[test_split])

        ranks = list()
        for batch_id, batch in enumerate(batches, 1):
            logger.debug(" DistMult test batch {} / {}".format(batch_id,
                                                               nbatches))
            batch_data = test_data[batch]

            h = batch_data[:, 0]
            r = batch_data[:, 1]
            t = batch_data[:, 2]

            ranks_hr = compute_ranks(h, r, t, node_embeddings, edge_embeddings)
            ranks_rt = compute_ranks(t, r, h, node_embeddings, edge_embeddings)
            ranks.append(torch.cat([ranks_hr, ranks_rt]))

        ranks = torch.cat(ranks)
        mrr += torch.mean(1.0 / ranks.float())
        for k in [1, 3, 10]:
            hits_at_k[k] += float(torch.mean((ranks <= k).float()))

    mrr = mrr / nbatches
    for k in hits_at_k.keys():
        hits_at_k[k] = hits_at_k[k] / nbatches

    logging.info("Performance on {} set: MRR (raw) {:.4f}".format(test_split, mrr)
                 + " / " + " / ".join(["H@{} {:.4f}".format(k,v) for
                                              k,v in hits_at_k.items()]))

    return (mrr, hits_at_k)

def build_dataset(kg, nodes_map, config, featureless):
    logger.debug("Starting dataset build")
    if featureless:
        F = dict()
    else:
        separate_literals = config['graph']['structural']['separate_literals']
        F = construct_features(nodes_map, kg,
                               config['graph']['features'],
                               separate_literals)

    Y = torch.empty((0, 0), dtype=torch.float16)  # dummy

    logger.debug("Completed dataset build")

    return (F, Y)

def build_model(C, A, modules_config, config, featureless):
    layers = config['model']['layers']
    logger.debug("Starting model build")

    # get sizes from dataset
    X_dim = C  # == 0 if featureless
    num_nodes = A.shape[0]
    num_relations = int(A.size()[1]/num_nodes)

    modules = list()
    # input layer
    modules.append((X_dim,
                    layers[0]['hidden_nodes'],
                    layers[0]['type'],
                    nn.ReLU()))

    # intermediate layers (if any)
    i = 1
    for layer in layers[1:-1]:
        modules.append((layers[i-1]['hidden_nodes'],
                        layer['hidden_nodes'],
                        layers[i-1]['type'],
                        nn.ReLU()))

        i += 1

    model = MRGCN(modules, modules_config, num_relations, num_nodes,
                  num_bases=config['model']['num_bases'],
                  p_dropout=config['model']['p_dropout'],
                  featureless=featureless,
                  bias=config['model']['bias'],
                  link_prediction=True)

    logger.debug("Completed model build")

    return model

def binary_crossentropy(Y_hat, Y, criterion):
    # Y_hat := output of score()
    # Y := labels in [0, 1]
    # Y_hat[i] == Y[i] -> i is same triple
    return criterion(Y_hat, Y)

def score_distmult(s, p, o):
    # s := vector embeddings of subjects
    # p := vector embeddings (diag of matrix) of predicates
    # o := vectpr embeddings of objects
    return torch.sum(s * p * o, dim=1)

def compute_ranks(e, r, targets, node_embeddings, edge_embeddings):
    partial_score = node_embeddings[e] * edge_embeddings[r]

    # compute score for each node as target
    partial_score = torch.transpose(partial_score, 0, 1).unsqueeze(2)
    node_repr = torch.transpose(node_embeddings, 0, 1).unsqueeze(1)

    score = torch.sum(torch.bmm(partial_score, node_repr), dim=0)
    score = torch.sigmoid(score)

    # compute ranks
    node_idx = torch.sort(score, dim=1, descending=True)[1]
    ranks = torch.nonzero(node_idx == torch.as_tensor(targets).view(-1, 1))

    return ranks[:, 1] + 1  # set index start at 1
