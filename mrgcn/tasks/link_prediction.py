#!/usr/bin/python3

from copy import deepcopy
import logging
import os
from time import time, sleep

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from mrgcn.encodings.graph_features import construct_features
from mrgcn.models.mrgcn import MRGCN


logger = logging.getLogger(__name__)
try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass

def run(A, X, C, data, tsv_writer, device, config,
        modules_config, featureless, test_split):
    # evaluation
    filtered_ranks = config['task']['filter_ranks']
    multiprocessing = config['task']['multiprocessing']

    tsv_label = "filtered" if filtered_ranks else "raw"
    tsv_writer.writerow(["epoch", "train_loss",
                         "valid_mrr_raw", "valid_H@1", "valid_H@3", "valid_H@10",
                         "test_mrr_{}".format(tsv_label), "test_H@1", "test_H@3",
                         "test_H@10"])

    # compile model
    num_nodes = A.shape[0]
    model = build_model(C, A, modules_config, config, featureless)
    optimizer = optim.Adam(model.parameters(),
                           lr=config['model']['learning_rate'],
                           weight_decay=config['model']['l2norm'])
    criterion = nn.BCEWithLogitsLoss()

    # mini batching
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
                             criterion, nepoch, filtered_ranks,
                             mrr_batch_size, device):
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

    ## test model
    # Log wall-clock time
    t0 = time()

    mrr, hits_at_k, ranks = test_model(A, X, data, num_nodes, model, criterion,
                                       filtered_ranks, mrr_batch_size, test_split,
                                       multiprocessing, device)

    logging.info("Testing time: {:.2f}s".format(time()-t0))

    # log metrics
    tsv_writer.writerow(["-1", "-1", "-1", "-1", "-1", "-1",
                         str(mrr), str(hits_at_k[1]),
                         str(hits_at_k[3]), str(hits_at_k[10])])

    return (mrr, hits_at_k, ranks)

def train_model(A, X, data, num_nodes, model, optimizer, criterion,
                nepoch, filtered_ranks, mrr_batch_size,
                device):
    logging.info("Training for {} epoch".format(nepoch))

    nsamples = data['train'].shape[0]
    for epoch in range(1, nepoch+1):
        model.train()
        # Single iteration
        node_embeddings = model(X, A, epoch=epoch, device=device)
        edge_embeddings = model.rgcn.relations
        train_data = deepcopy(data["train"])

        # sample negative triples
        ncorrupt = nsamples//5
        neg_samples_idx = np.random.choice(np.arange(nsamples),
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
        Y = torch.ones(nsamples, dtype=torch.float32)
        Y[neg_samples_idx] = 0

        # compute score
        Y_hat = score_distmult(node_embeddings[train_data[:, 0]],
                               edge_embeddings[train_data[:, 1]],
                               node_embeddings[train_data[:, 2]])

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
            node_embeddings = model(X, A, epoch=-1, device=device)
            edge_embeddings = model.rgcn.relations

            ranks = compute_ranks(data,
                                  node_embeddings,
                                  edge_embeddings,
                                  "valid",
                                  mrr_batch_size,
                                  filtered=False)
            mrr_raw = torch.mean(1.0 / ranks.float()).item()

            hits_at_k = dict()
            for k in [1, 3, 10]:
                hits_at_k[k] = float(torch.mean((ranks <= k).float()))

        logging.info("{:04d} ".format(epoch) \
                     + "| train loss {:.4f} ".format(loss)
                     + "| valid MRR (raw) {:.4f} ".format(mrr_raw)
                     + "/ " + " / ".join(["H@{} {:.4f}".format(k,v)
                                         for k,v in hits_at_k.items()]))

        yield (epoch, loss, mrr_raw, hits_at_k[1], hits_at_k[3], hits_at_k[10])

def test_model(A, X, data, num_nodes, model, criterion, filtered_ranks,
               mrr_batch_size, test_split, multiprocessing, device):
    model.eval()
    mrr = 0.0
    hits_at_k = {1: 0.0, 3: 0.0, 10: 0.0}
    ranks = None
    with torch.no_grad():
        node_embeddings = model(X, A, epoch=-1, device=device)
        edge_embeddings = model.rgcn.relations

        ranks = compute_ranks(data,
                              node_embeddings,
                              edge_embeddings,
                              test_split,
                              mrr_batch_size,
                              filtered_ranks,
                              multiprocessing)

        mrr = torch.mean(1.0 / ranks.float()).item()
        for k in [1, 3, 10]:
            hits_at_k[k] = float(torch.mean((ranks <= k).float()))

    rank_type = "filtered" if filtered_ranks else "raw"
    logging.info("Performance on {} set: MRR ({}) {:.4f}".format(test_split,
                                                                 rank_type,
                                                                 mrr)
                 + " / " + " / ".join(["H@{} {:.4f}".format(k,v) for
                                              k,v in hits_at_k.items()]))

    return (mrr, hits_at_k, ranks.numpy())

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

def compute_ranks(data, node_embeddings, edge_embeddings, eval_split,
                  batch_size, filtered=False, multiprocessing=False):
    if filtered:
        if multiprocessing:
            return compute_ranks_filtered_mp(data, node_embeddings, edge_embeddings,
                                   eval_split)
        else:
            return compute_ranks_filtered(data, node_embeddings, edge_embeddings,
                                   eval_split)
    else:
        return compute_ranks_raw(data, node_embeddings, edge_embeddings, eval_split,
                          batch_size)

def compute_ranks_filtered(data, node_embeddings, edge_embeddings, eval_split):
    num_nodes = node_embeddings.shape[0]

    # don't blacklist the samples we use to evaluate
    if eval_split == 'test':
        blacklist = torch.from_numpy(np.concatenate([data['train'], data['valid']],
                                                    axis=0))
    else:
        blacklist = torch.from_numpy(np.concatenate([data['train'], data['test']],
                                                    axis=0))

    entities = torch.arange(num_nodes).unsqueeze(1)
    ranks = list()
    for h,r,t in data[eval_split]:
        included_h = construct_filtered_head(entities, blacklist,
                                                h, r, t)
        included_t = construct_filtered_tail(entities, blacklist,
                                                h, r, t)
        for e, u, included in zip((h, t),
                                  (t, h),
                                  (included_h, included_t)):
            score = score_distmult(node_embeddings[included],
                                   edge_embeddings[r],
                                   node_embeddings[u])
            score = torch.sigmoid(score)

            # compute ranks
            e_idx = (included == e).nonzero(as_tuple=True)[0].item()
            node_idx = torch.sort(score, descending=True)[1]
            rank = (node_idx == e_idx).nonzero(as_tuple=True)[0].item()

            ranks.append(rank)

    return torch.LongTensor(ranks) + 1 # set index start at 1

def compute_ranks_filtered_mp(data, node_embeddings, edge_embeddings, eval_split):
    num_nodes = node_embeddings.shape[0]
    entities = torch.arange(num_nodes).unsqueeze(1)
    nproc = len(os.sched_getaffinity(0))

    # don't blacklist the samples we use to evaluate
    if eval_split == 'test':
        blacklist = torch.from_numpy(np.concatenate([data['train'], data['valid']],
                                                    axis=0))
    else:
        blacklist = torch.from_numpy(np.concatenate([data['train'], data['test']],
                                                    axis=0))

    # create jobs
    nsamples = data[eval_split].shape[0]
    batches = np.array_split(np.arange(data[eval_split].shape[0]),
                             min(nsamples, nproc))
    jobs = [(data[eval_split][batch], entities, blacklist,
            node_embeddings, edge_embeddings) for batch in batches]

    ranks = list()
    with mp.Pool(processes=nproc) as pool:
        logger.debug(" Computing ranks with {} workers".format(nproc))
        work = pool.map_async(compute_ranks_filtered_mp_worker, jobs,
                              callback=ranks.extend)

        nchunks = work._number_left
        current_chunk = 0
        while True:
            sleep(0.5)
            chunks_done = nchunks - work._number_left
            if chunks_done > current_chunk:
                current_chunk = chunks_done
                logger.debug(" Processed {} / {} chunks".format(chunks_done,
                                                                nchunks))
            if work.ready():
                break

    return torch.LongTensor(ranks) + 1 # set index start at 1

def compute_ranks_filtered_mp_worker(inputs):
    # return list rather than separate ranks to reduce messages
    batch, entities, blacklist, node_embeddings, edge_embeddings = inputs

    ranks = list()
    for h,r,t in batch:
        included_h = construct_filtered_head(entities, blacklist,
                                                h, r, t)
        included_t = construct_filtered_tail(entities, blacklist,
                                                h, r, t)
        for e, u, included in zip((h, t),
                                  (t, h),
                                  (included_h, included_t)):
            score = score_distmult(node_embeddings[included],
                                   edge_embeddings[r],
                                   node_embeddings[u])
            score = torch.sigmoid(score)

            # compute ranks
            e_idx = (included == e).nonzero(as_tuple=True)[0].item()
            node_idx = torch.sort(score, descending=True)[1]
            rank = (node_idx == e_idx).nonzero(as_tuple=True)[0].item()

            ranks.append(rank)

    return ranks

def construct_filtered_head(entities, triples, h, r, t):
    num_nodes = entities.shape[0]

    # (num_nodes, 3) tensor of triples we want to filter
    excluded = torch.tensor([r,t]).repeat_interleave(num_nodes)
    excluded = excluded.reshape((2, num_nodes)).T
    excluded = torch.cat([entities, excluded.long()], dim=1)

    # indices of triples that are not in excluded
    return (~(excluded[:, None] == triples).all(-1).any(-1)).nonzero(as_tuple=True)[0]

def construct_filtered_tail(entities, triples, h, r, t):
    num_nodes = entities.shape[0]

    # (num_nodes, 3) tensor of triples we want to filter
    excluded = torch.tensor([h,r]).repeat_interleave(num_nodes)
    excluded = excluded.reshape((2, num_nodes)).T
    excluded = torch.cat([excluded.long(), entities], dim=1)

    # indices of triples that are not in excluded
    return (~(excluded[:, None] == triples).all(-1).any(-1)).nonzero(as_tuple=True)[0]

def compute_ranks_raw(data, node_embeddings, edge_embeddings, eval_split,
                      batch_size):
    nsamples = data[eval_split].shape[0]
    batches = np.array_split(np.arange(nsamples),
                             max(1, nsamples//batch_size))
    nbatches = len(batches)
    ranks = list()
    for batch_id, batch in enumerate(batches, 1):
        logger.debug(" DistMult {} batch {} / {}".format(eval_split,
                                                         batch_id,
                                                         nbatches))
        batch_data = data[eval_split][batch]

        h = batch_data[:, 0]
        r = batch_data[:, 1]
        t = batch_data[:, 2]

        ranks_hr = compute_ranks_raw_one_side(h, r, t, node_embeddings, edge_embeddings)
        ranks_rt = compute_ranks_raw_one_side(t, r, h, node_embeddings, edge_embeddings)
        ranks.append(torch.cat([ranks_hr, ranks_rt]))

    return torch.cat(ranks) + 1 # set index start at 1

def compute_ranks_raw_one_side(e, r, targets, node_embeddings, edge_embeddings):
    partial_score = node_embeddings[e] * edge_embeddings[r]

    # compute score for each node as target
    partial_score = torch.transpose(partial_score, 0, 1).unsqueeze(2)
    node_repr = torch.transpose(node_embeddings, 0, 1).unsqueeze(1)

    score = torch.sum(torch.bmm(partial_score, node_repr), dim=0)
    score = torch.sigmoid(score)

    # compute ranks
    node_idx = torch.sort(score, dim=1, descending=True)[1]
    ranks = torch.nonzero(node_idx == torch.as_tensor(targets).view(-1, 1),
                          as_tuple=True)[1]

    return ranks
