#!/usr/bin/python3

import logging
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mrgcn.encodings.graph_features import construct_features
from mrgcn.models.mrgcn import MRGCN


logger = logging.getLogger(__name__)

def run(A, X, C, data, splits, tsv_writer, device, config,
        modules_config, featureless, test_split):
    # evaluation
    filtered_ranks = config['task']['filter_ranks']

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
    mrr_batch_size = int(config['model']['mrr_batch_size'])

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
    for epoch in train_model(A, X, data, splits, num_nodes, model, optimizer,
                             criterion, nepoch, mrr_batch_size, device):
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

    mrr, hits_at_k, ranks = test_model(A, X, data, splits, num_nodes, model, criterion,
                                       filtered_ranks, mrr_batch_size, test_split,
                                       device)

    logging.info("Testing time: {:.2f}s".format(time()-t0))

    # log metrics
    tsv_writer.writerow(["-1", "-1", "-1", "-1", "-1", "-1",
                         str(mrr), str(hits_at_k[1]),
                         str(hits_at_k[3]), str(hits_at_k[10])])

    return (mrr, hits_at_k, ranks)

def train_model(A, X, data, splits, num_nodes, model, optimizer, criterion,
                nepoch, mrr_batch_size, device):
    logging.info("Training for {} epoch".format(nepoch))

    idx_begin, idx_end = splits['train'][0], splits['train'][1]
    train_data = data[idx_begin:idx_end]
    nsamples = train_data.shape[0]
    for epoch in range(1, nepoch+1):
        model.train()
        # Single iteration
        node_embeddings = model(X, A, epoch=epoch, device=device)
        edge_embeddings = model.rgcn.relations

        # sample negative triples by copying and corrupting positive triples
        ncorrupt = int(nsamples//5)
        neg_samples_idx = np.random.choice(np.arange(nsamples),
                                           ncorrupt,
                                           replace=False)

        ncorrupt_head = int(ncorrupt//2)
        ncorrupt_tail = ncorrupt - ncorrupt_head
        corrupted_data = torch.empty((ncorrupt, 3), dtype=torch.int64)

        corrupted_data = train_data[neg_samples_idx]
        corrupted_data[:ncorrupt_head, 0] = torch.from_numpy(np.random.choice(np.arange(num_nodes),
                                                                              ncorrupt_head))
        corrupted_data[-ncorrupt_tail:, 2] = torch.from_numpy(np.random.choice(np.arange(num_nodes),
                                                                               ncorrupt_tail))

        # create labels; positive samples are 1, negative 0
        Y = torch.ones(nsamples+ncorrupt, dtype=torch.float32)
        Y[-ncorrupt:] = 0

        # compute score
        Y_hat = torch.empty((nsamples+ncorrupt), dtype=torch.float32)
        Y_hat[:nsamples] = score_distmult(node_embeddings[train_data[:, 0]],
                                          edge_embeddings[train_data[:, 1]],
                                          node_embeddings[train_data[:, 2]])
        Y_hat[-ncorrupt:] = score_distmult(node_embeddings[corrupted_data[:, 0]],
                                           edge_embeddings[corrupted_data[:, 1]],
                                           node_embeddings[corrupted_data[:, 2]])

        # compute loss
        loss = binary_crossentropy(Y_hat, Y, criterion)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()  # training loss
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss = float(loss)  # remove pointer to gradients to free memory

        # validate
        model.eval()
        with torch.no_grad():
            node_embeddings = model(X, A, epoch=-1, device=device)
            edge_embeddings = model.rgcn.relations

            ranks = compute_ranks_fast(data,
                                      splits,
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

def test_model(A, X, data, splits, num_nodes, model, criterion, filtered_ranks,
               mrr_batch_size, test_split, device):
    model.eval()
    mrr = 0.0
    hits_at_k = {1: 0.0, 3: 0.0, 10: 0.0}
    ranks = None
    with torch.no_grad():
        node_embeddings = model(X, A, epoch=-1, device=device)
        edge_embeddings = model.rgcn.relations

        ranks = compute_ranks_fast(data,
                                  splits,
                                  node_embeddings,
                                  edge_embeddings,
                                  test_split,
                                  mrr_batch_size,
                                  filtered_ranks)

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

#def compute_ranks(data, splits, node_embeddings, edge_embeddings, eval_split,
#                  batch_size_raw, batch_size_filtered, filtered=False):
#    if filtered:
#        return compute_ranks_filtered(data, splits, node_embeddings, edge_embeddings,
#                               eval_split, batch_size_filtered)
#    else:
#        return compute_ranks_raw(data, splits, node_embeddings, edge_embeddings, eval_split,
#                          batch_size_raw)

#def compute_ranks_filtered(data, splits, node_embeddings, edge_embeddings, eval_split,
#                           batch_size=100):
#    idx_begin, idx_end = splits[eval_split][0], splits[eval_split][1]
#    nsamples = idx_end - idx_begin
#    ranks = torch.empty((nsamples*2), dtype=torch.int64)
#    i = 0
#    for h,r,t in data[idx_begin:idx_end]:
#        ranks[i] = compute_ranks_filtered_one_side(data,
#                                                   node_embeddings,
#                                                   edge_embeddings,
#                                                   r, h, t, batch_size,
#                                                   perturb='head')
#        ranks[i+1] = compute_ranks_filtered_one_side(data,
#                                                     node_embeddings,
#                                                     edge_embeddings,
#                                                     r, t, h, batch_size,
#                                                     perturb='tail')
#
#        i += 2
#
#    return ranks + 1 # set index start at 1
#
#def compute_ranks_filtered_one_side(data, node_embeddings, edge_embeddings,
#                                    r, e, u, batch_size, perturb):
#    # target triple
#    target_idx = 0
#    target_fact = torch.cat([node_embeddings[e] *
#                             edge_embeddings[r] *
#                             node_embeddings[u]])
#    score = [torch.sum(target_fact.unsqueeze(1).T, dim=1)]  # distmult
#
#    # compute scores for non-existing triples
#    num_nodes = node_embeddings.shape[0]
#    num_batches = int((num_nodes + batch_size-1)//batch_size)
#    for batch_id in range(num_batches):
#        batch_begin = batch_id * batch_size
#        batch_end = min(num_nodes, (batch_id+1) * batch_size)
#        batch_entities = torch.arange(batch_begin, batch_end).unsqueeze(1)
#
#        included = batch_begin + construct_filtered_set(batch_entities,
#                                                        data,
#                                                        r, u, perturb)
#
#        score.append(score_distmult(node_embeddings[included],
#                                    edge_embeddings[r],
#                                    node_embeddings[u]))
#
#    score = torch.sigmoid(torch.cat(score))
#
#    # compute ranks
#    _, node_idx = torch.sort(score, descending=True)
#    rank = (node_idx == target_idx).nonzero(as_tuple=True)[0].item()
#
#    return rank
#
#def construct_filtered_set(entities, data, r, u, perturb):
#    num_nodes = entities.shape[0]
#
#    if perturb == 'head':
#        # (num_nodes, 3) tensor of triples we want to filter
#        excluded = torch.tensor([r,u]).repeat_interleave(num_nodes)
#        excluded = excluded.reshape((2, num_nodes)).T
#        excluded = torch.cat([entities, excluded.long()], dim=1)
#    else:  # 'tail'
#        # (num_nodes, 3) tensor of triples we want to filter
#        excluded = torch.tensor([u,r]).repeat_interleave(num_nodes)
#        excluded = excluded.reshape((2, num_nodes)).T
#        excluded = torch.cat([excluded.long(), entities], dim=1)
#
#    # indices of triples that are not in excluded
#    return (~(excluded[:, None] == data).all(-1).any(-1)).nonzero(as_tuple=True)[0]
#
#def compute_ranks_raw(data, splits, node_embeddings, edge_embeddings, eval_split,
#                      batch_size):
#    idx_begin, idx_end = splits[eval_split][0], splits[eval_split][1]
#    eval_set = data[idx_begin:idx_end]
#
#    num_samples = eval_set.shape[0]
#    num_batches = int((num_samples + batch_size-1)//batch_size)
#    ranks = torch.empty((num_samples*2), dtype=torch.int64)
#    for batch_id in range(num_batches):
#        batch_begin = batch_id * batch_size
#        batch_end = min(num_samples, (batch_id+1) * batch_size)
#        logger.debug(" DistMult {} batch {} / {}".format(eval_split,
#                                                         batch_id+1,
#                                                         num_batches))
#        batch_data = eval_set[batch_begin:batch_end]
#
#        h = batch_data[:, 0]
#        r = batch_data[:, 1]
#        t = batch_data[:, 2]
#
#        ranks[batch_begin:batch_end] = compute_ranks_raw_one_side(h, r, t,
#                                                                  node_embeddings,
#                                                                  edge_embeddings)
#        ranks[num_samples+batch_begin:num_samples+batch_end] = compute_ranks_raw_one_side(t, r, h,
#                                                                                          node_embeddings,
#                                                                                          edge_embeddings)
#
#    return ranks + 1 # set index start at 1
#
#def compute_ranks_raw_one_side(e, r, targets, node_embeddings, edge_embeddings):
#    partial_score = node_embeddings[e] * edge_embeddings[r]
#
#    # compute score for each node as target
#    partial_score = torch.transpose(partial_score, 0, 1).unsqueeze(2)
#    node_repr = torch.transpose(node_embeddings, 0, 1).unsqueeze(1)
#
#    score = torch.sum(torch.bmm(partial_score, node_repr), dim=0)
#    score = torch.sigmoid(score)
#
#    # compute ranks
#    node_idx = torch.sort(score, dim=1, descending=True)[1]
#    ranks = torch.nonzero(node_idx == torch.as_tensor(targets).view(-1, 1),
#                          as_tuple=True)[1]
#
#    return ranks

#################

def filter_scores_(scores, batch_data, heads, tails, head=True):
     # set scores of existing facts to -inf
    indices = list()
    for i, (s, p, o) in enumerate(batch_data):
        s, p, o = (s.item(), p.item(), o.item())
        if head:
            indices.extend([(i, si) for si in heads[p, o] if si != s])
        else:
            indices.extend([(i, oi) for oi in tails[s, p] if oi != o])
        # we add the indices of all know triples except the one corresponding
        # to the target triples.

    indices = torch.tensor(indices)
    scores[indices[:, 0], indices[:, 1]] = float('-inf')

def truedicts(facts):
    heads = dict()
    tails = dict()
    for i in range(facts.shape[0]):
        fact = facts[i]
        s, p, o = fact[0].item(), fact[1].item(), fact[2].item()

        if (p, o) not in heads.keys():
            heads[(p, o)] = list()
        if (s, p) not in tails.keys():
            tails[(s, p)] = list()

        heads[(p, o)].append(s)
        tails[(s, p)].append(o)

    return heads, tails

def compute_ranks_fast(data, splits, node_embeddings, edge_embeddings, eval_split,
                       batch_size=16, filtered=True):
    idx_begin, idx_end = splits[eval_split][0], splits[eval_split][1]
    eval_set = data[idx_begin:idx_end]

    true_heads, true_tails = truedicts(data) if filtered else (None, None)

    num_facts = eval_set.shape[0]
    num_nodes = node_embeddings.shape[0]
    num_batches = int((num_facts + batch_size-1)//batch_size)
    ranks = torch.empty((num_facts*2), dtype=torch.int64)
    for head in [True, False]:  # head or tail prediction
        offset = int(head) * num_facts
        for batch_id in range(num_batches):
            batch_begin = batch_id * batch_size
            batch_end = min(num_facts, (batch_id+1) * batch_size)

            batch_data = eval_set[batch_begin:batch_end]
            batch_num_facts = batch_data.shape[0]

            # compute the full score matrix (filter later)
            bases   = batch_data[:, 1:] if head else batch_data[:, :2]
            targets = batch_data[:, 0]  if head else batch_data[:, 2]

            # collect the triples for which to compute scores
            bexp = bases.view(batch_num_facts, 1, 2).expand(batch_num_facts,
                                                            num_nodes, 2)
            ar   = torch.arange(num_nodes).view(1, num_nodes, 1).expand(batch_num_facts,
                                                                        num_nodes, 1)
            candidates = torch.cat([ar, bexp] if head else [bexp, ar], dim=2)

            scores = score_distmult_bc(candidates[:,:,0], candidates[:,:,1], candidates[:,:,2],
                                       node_embeddings, edge_embeddings)

            # filter out the true triples that aren't the target
            if filtered:
                filter_scores_(scores, batch_data, true_heads, true_tails, head=head)

            # Select the true scores, and count the number of values larger than than
            true_scores = scores[torch.arange(batch_num_facts), targets]
            batch_ranks = torch.sum(scores > true_scores.view(batch_num_facts, 1), dim=1, dtype=torch.int64)
            # -- This is the "optimistic" rank (assuming it's sorted to the front of the ties)
            num_ties = torch.sum(scores == true_scores.view(batch_num_facts, 1), dim=1, dtype=torch.int64)

            # Account for ties (put the true example halfway down the ties)
            batch_ranks = batch_ranks + (num_ties - 1) // 2

            ranks[offset+batch_begin:offset+batch_end] = batch_ranks

    return ranks + 1

def score_distmult_bc(si, pi, oi, node_embeddings, edge_embeddings):
    s = node_embeddings[si, :]
    p = edge_embeddings[pi, :]
    o = node_embeddings[oi, :]

    if len(s.size()) == len(p.size()) == len(o.size()): # optimizations for common broadcasting
        if pi.size(-1) == 1 and oi.size(-1) == 1:
            singles = p * o # ignoring batch dimensions, this is a single vector
            return torch.matmul(s, singles.transpose(-1, -2)).squeeze(-1)

        if si.size(-1) == 1 and oi.size(-1) == 1:
            singles = s * o
            return torch.matmul(p, singles.transpose(-1, -2)).squeeze(-1)

        if si.size(-1) == 1 and pi.size(-1) == 1:
            singles = s * p
            return torch.matmul(o, singles.transpose(-1, -2)).squeeze(-1)

    return torch.sum(s * p * o, dim = -1)
