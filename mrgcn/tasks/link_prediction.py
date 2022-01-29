#!/usr/bin/python3

import logging
from math import ceil
from shutil import get_terminal_size
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mrgcn.encodings.graph_features import construct_features
from mrgcn.models.mrgcn import MRGCN
from mrgcn.tasks.utils import optimizer_params


logger = logging.getLogger(__name__)


def run(A, X, X_width, data, tsv_writer, model_device, distmult_device,
        config, modules_config, optimizer_config, featureless, test_split):
    header = ["epoch", "loss"]
    for split in ["train", "valid", "test"]:
        header.extend([split+"_mrr_raw", split+"_H@1_raw", split+"_H@3_raw",
                       split+"_H@10_raw", split+"_mrr_flt", split+"_H@1_flt",
                       split+"_H@3_flt", split+"_H@10_flt"])
    tsv_writer.writerow(header)

    # used for clearing a line
    term_width = get_terminal_size().columns

    # compile model
    num_nodes = A.shape[0]
    model = build_model(X_width, A, modules_config, config, featureless)
    opt_params = optimizer_params(model, optimizer_config)
    optimizer = optim.Adam(opt_params,
                           lr=config['model']['learning_rate'],
                           weight_decay=config['model']['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()

    # mini batching
    mrr_batchsize = int(config['task']['mrr_batchsize'])
    batchsize = config['task']['batchsize']

    # train model
    nepoch = config['model']['epoch']
    eval_interval = config['task']['eval_interval']
    filter_ranks = config['task']['filter_ranks']
    l1_lambda = config['model']['l1_lambda']
    l2_lambda = config['model']['l2_lambda']

    # store nodes needed per batch; used as pointer
    batchsize_lst = list()

    # Log wall-clock time
    t0 = time()
    for result in train_model(A, X, data, num_nodes, model, optimizer,
                              criterion, nepoch, batchsize, batchsize_lst,
                              mrr_batchsize, eval_interval, filter_ranks,
                              l1_lambda, l2_lambda, model_device,
                              distmult_device, term_width):

        epoch, loss, train_mrr, train_hits_at_k,\
                     valid_mrr, valid_hits_at_k = result

        result_str = [str(epoch), str(loss)]
        for mrr, hits in [(train_mrr, train_hits_at_k),
                          (valid_mrr, valid_hits_at_k)]:
            if mrr is None or hits is None:
                result_str.extend([-1, -1, -1, -1, -1, -1, -1, -1])
                continue

            result_str.extend([str(mrr['raw']), str(hits['raw'][0]),
                               str(hits['raw'][1]), str(hits['raw'][2]),
                               str(mrr['flt']), str(hits['flt'][0]),
                               str(hits['flt'][1]), str(hits['flt'][2])])

        # add test set placeholder
        result_str.extend([-1, -1, -1, -1, -1, -1, -1, -1])

        # log metrics
        tsv_writer.writerow(result_str)

    logging.info("Training time: {:.2f}s".format(time()-t0))

    # the highest seen batchsize during training
    node_batchsize = max(batchsize_lst)

    # Log wall-clock time
    t0 = time()

    test_data = data[test_split]
    test_mrr, test_hits_at_k, test_ranks, _ = test_model(A, X, test_data,
                                                         model, node_batchsize,
                                                         mrr_batchsize,
                                                         filter_ranks,
                                                         model_device,
                                                         distmult_device,
                                                         term_width)

    logging.info("Testing time: {:.2f}s".format(time()-t0))

    # log metrics
    result_str = [-1 for _ in range(18)]
    result_str.extend([str(test_mrr['raw']),
                       str(test_hits_at_k['raw'][0]),
                       str(test_hits_at_k['raw'][1]),
                       str(test_hits_at_k['raw'][2]),
                       str(test_mrr['flt']),
                       str(test_hits_at_k['flt'][0]),
                       str(test_hits_at_k['flt'][1]),
                       str(test_hits_at_k['flt'][2])])
    tsv_writer.writerow(result_str)

    return (test_mrr, test_hits_at_k, test_ranks)


#def train_model(A, X, data, num_nodes, model, optimizer, criterion,
#                nepoch, mrr_batch_size, eval_interval, batchsize, 
#                filter_ranks, l1_lambda, l2_lambda, model_device,
#                distmult_device):
#    logging.info("Training for {} epoch".format(nepoch))
#
#    train_data = data["train"]
#    nsamples = train_data.shape[0]
#    for epoch in range(1, nepoch+1):
#        model.train()
#        # Single iteration
#        node_embeddings = model(X, A, epoch=epoch,
#                                device=model_device).to(distmult_device)
#        edge_embeddings = model.rgcn.relations.to(distmult_device)
#
#        # sample negative triples by copying and corrupting positive triples
#        ncorrupt = nsamples//5
#        neg_samples_idx = torch.from_numpy(np.random.choice(np.arange(nsamples),
#                                                            ncorrupt,
#                                                            replace=False))
#
#        ncorrupt_head = ncorrupt//2
#        ncorrupt_tail = ncorrupt - ncorrupt_head
#        corrupted_data = torch.empty((ncorrupt, 3), dtype=torch.int64)
#
#        corrupted_data = train_data[neg_samples_idx]
#        corrupted_data[:ncorrupt_head, 0] = torch.from_numpy(np.random.choice(np.arange(num_nodes),
#                                                                              ncorrupt_head))
#        corrupted_data[-ncorrupt_tail:, 2] = torch.from_numpy(np.random.choice(np.arange(num_nodes),
#                                                                               ncorrupt_tail))
#
#        # create labels; positive samples are 1, negative 0
#        Y = torch.ones(nsamples+ncorrupt, dtype=torch.float32)
#        Y[-ncorrupt:] = 0
#
#        # compute score
#        Y_hat = torch.empty((nsamples+ncorrupt), dtype=torch.float32)
#        Y_hat[:nsamples] = score_distmult_bc((train_data[:, 0],
#                                              train_data[:, 1],
#                                              train_data[:, 2]),
#                                             node_embeddings,
#                                             edge_embeddings).to('cpu')
#        Y_hat[-ncorrupt:] = score_distmult_bc((corrupted_data[:, 0],
#                                               corrupted_data[:, 1],
#                                               corrupted_data[:, 2]),
#                                              node_embeddings,
#                                              edge_embeddings).to('cpu')
#
#        # clear gpu cache to save memory
#        if model_device == torch.device('cpu') and\
#           distmult_device != torch.device('cpu'):
#            del node_embeddings
#            del edge_embeddings
#            torch.cuda.empty_cache()
#
#        # compute loss
#        optimizer.zero_grad()
#        loss = binary_crossentropy(Y_hat, Y, criterion)

def train_model(A, X, data, num_nodes, model, optimizer, criterion,
                nepoch, batchsize, batchsize_lst, mrr_batchsize,
                eval_interval, filter_ranks, l1_lambda, l2_lambda,
                model_device, distmult_device, term_width):
    logging.info("Training for {} epoch".format(nepoch))

    train_data = data["train"]
    num_samples = train_data.shape[0]
    
    if batchsize <= 0:
        batchsize = num_samples

    batches = [slice(begin, min(begin+batchsize, num_samples))
               for begin in range(0, num_samples, batchsize)]
    num_batches = len(batches)
    for epoch in range(1, nepoch+1):
        model.train()
        
        loss_lst = list()
        for batch_id, batch in enumerate(batches, 1):
            batch_str = " [TRAIN] - batch %2.d / %d" % (batch_id, num_batches)
            print(batch_str, end='\b'*len(batch_str), flush=True)

            if num_batches > 1:
                batch_data = train_data[batch].detach().clone()
                batch_nodes = np.union1d(batch_data[:, 0], batch_data[:, 2])
            else:
                # full batch
                batch_data = train_data
                batch_nodes = np.arange(num_nodes)

            batch_num_samples = batch_data.shape[0]
            batch_num_nodes = len(batch_nodes)

            # sample negative triples by copying and corrupting positive triples
            ncorrupt = batch_num_samples//5
            neg_samples_idx = torch.from_numpy(
                np.random.choice(np.arange(batch_num_samples),
                                 ncorrupt,
                                 replace=False))

            ncorrupt_head = ncorrupt//2
            ncorrupt_tail = ncorrupt - ncorrupt_head
            corrupted_data = torch.empty((ncorrupt, 3), dtype=torch.int64)

            # within-batch corruption to reduce the need to compute others
            corrupted_data = batch_data[neg_samples_idx]
            corrupted_data[:ncorrupt_head, 0] = torch.from_numpy(np.random.choice(batch_nodes,
                                                                                  ncorrupt_head))
            corrupted_data[-ncorrupt_tail:, 2] = torch.from_numpy(np.random.choice(batch_nodes,
                                                                                   ncorrupt_tail))

            # remap nodes to match embedding index
            if num_batches > 1:
                node_idx_map = {i:j for j,i in enumerate(batch_nodes)}
                batch_data[:, 0] = torch.LongTensor([node_idx_map[int(i)]
                                                     for i in batch_data[:, 0]])
                batch_data[:, 2] = torch.LongTensor([node_idx_map[int(i)]
                                                     for i in batch_data[:, 2]])
                corrupted_data[:, 0] = torch.LongTensor([node_idx_map[int(i)]
                                                         for i in corrupted_data[:, 0]])
                corrupted_data[:, 2] = torch.LongTensor([node_idx_map[int(i)]
                                                         for i in corrupted_data[:, 2]])

            # create labels; positive samples are 1, negative 0
            Y = torch.ones(batch_num_samples+ncorrupt, dtype=torch.float32)
            Y[-ncorrupt:] = 0

            # compute score
            node_embeddings = model(X, A, batch_nodes,
                                    device=model_device).to(distmult_device)
            edge_embeddings = model.rgcn.relations.to(distmult_device)

            # subset of nodes in batch (rest are neighbours)
            node_embeddings = node_embeddings[:batch_num_nodes]

            Y_hat = torch.empty((batch_num_samples+ncorrupt), dtype=torch.float32)
            Y_hat[:batch_num_samples] = score_distmult_bc((batch_data[:, 0],
                                                           batch_data[:, 1],
                                                           batch_data[:, 2]),
                                                          node_embeddings,
                                                          edge_embeddings).to('cpu')
            Y_hat[-ncorrupt:] = score_distmult_bc((corrupted_data[:, 0],
                                                   corrupted_data[:, 1],
                                                   corrupted_data[:, 2]),
                                                  node_embeddings,
                                                  edge_embeddings).to('cpu')

            # clear gpu cache to save memory
            if model_device == torch.device('cpu') and\
               distmult_device != torch.device('cpu'):
                del node_embeddings
                del edge_embeddings
                torch.cuda.empty_cache()

            # compute loss
            optimizer.zero_grad()
            loss = binary_crossentropy(Y_hat, Y, criterion)

            if l1_lambda > 0:
                l1_regularization = torch.tensor(0.)
                for name, param in model.named_parameters():
                    if 'weight' not in name:
                        continue
                    l1_regularization += torch.sum(param.abs())

                loss += l1_lambda * l1_regularization

            if l2_lambda > 0:
                l2_regularization = torch.tensor(0.)
                for name, param in model.named_parameters():
                    if 'weight' not in name:
                        continue
                    l2_regularization += torch.sum(param ** 2)

                loss += l2_lambda * l2_regularization

            loss.backward()  # training loss
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_lst.append(float(loss))
            batchsize_lst.append(batch_num_nodes)
        
        loss = np.mean(loss_lst)
        results_str = f"{epoch:04d} | loss {loss:.4f}"

        train_mrr, train_hits_at_k = None, None
        valid_mrr, valid_hits_at_k = None, None
        if epoch % eval_interval == 0 or epoch == nepoch:
            # the highest seen batchsize during training
            node_batchsize = max(batchsize_lst)
            logger.debug("Using %d nodes per batch for evaluation" % node_batchsize)

            train_mrr, train_hits_at_k, _, cache = test_model(A, X, train_data, model,
                                                              node_batchsize,
                                                              mrr_batchsize,
                                                              filter_ranks,
                                                              model_device,
                                                              distmult_device,
                                                              term_width)

            results_str += f" | train MRR {train_mrr['raw']:.4f} (raw)"
            if filter_ranks:
                results_str += f" / {train_mrr['flt']:.4f} (filtered)"

            valid_mrr = None
            valid_hits_at_k = None
            if "valid" in data.keys() and epoch < nepoch:
                valid_data = data["valid"]
                valid_mrr, valid_hits_at_k, _, _ = test_model(A, X, valid_data,
                                                              model,
                                                              node_batchsize,
                                                              mrr_batchsize,
                                                              filter_ranks,
                                                              model_device,
                                                              distmult_device,
                                                              term_width,
                                                              cache)

                results_str += f" | valid MRR {valid_mrr['raw']:.4f} (raw) "
                if filter_ranks:
                    results_str += f" / flt {valid_mrr['flt']:.4f} (filtered)"
        else:
            width_remain = term_width - len(results_str)
            results_str += " " * (width_remain//2)  # ensure overwrite of batch message

        logging.info(results_str)

        yield (epoch, loss,
               train_mrr, train_hits_at_k,
               valid_mrr, valid_hits_at_k)


def test_model(A, X, data, model, node_batchsize, mrr_batch_size, filter_ranks,
               model_device, distmult_device, term_width, cache=None):
    model.eval()
    
    num_nodes = A.shape[0]
    if node_batchsize <= 0:
        node_batchsize = num_nodes

    batches = [slice(begin, min(begin+node_batchsize, num_nodes))
               for begin in range(0, num_nodes, node_batchsize)]
    num_batches = len(batches)
    
    mrr = dict()
    hits_at_k = dict()
    rankings = dict()
    with torch.no_grad():
        if cache is not None:
            node_embeddings, edge_embeddings = cache
        else:
            out_dim = model.rgcn.layers['layer_0'].outdim  # assume one GCN layer
            node_embeddings = torch.zeros((num_nodes, out_dim)) 
            for batch_id, batch in enumerate(batches, 1):
                batch_str = " [MRGCN] - batch %2.d / %d" % (batch_id, num_batches)
                width_remain = term_width - len(batch_str)
                batch_str += " " * (width_remain//2)  # ensure overwrite of batch message
                print(batch_str, end='\b'*len(batch_str), flush=True)

                batch_idx = np.arange(batch.start, batch.stop)
                batch_num_samples = len(batch_idx)
                batch_embeddings = model(X, A, batch_idx,
                                         device=model_device)[:batch_num_samples].to('cpu')

                node_embeddings[batch_idx, :] = batch_embeddings

            node_embeddings = node_embeddings.to(distmult_device)
            edge_embeddings = model.rgcn.relations.to(distmult_device)

        for filtered in [False, True]:
            rank_type = "flt" if filtered else "raw"
            if filtered is True and not filter_ranks:
                mrr[rank_type] = -1
                hits_at_k[rank_type] = [-1, -1, -1]
                rankings[rank_type] = [-1]

                continue

            ranks = compute_ranks_fast(data,
                                       node_embeddings,
                                       edge_embeddings,
                                       mrr_batch_size,
                                       filtered,
                                       term_width)

            mrr[rank_type] = torch.mean(1.0 / ranks.float()).item()
            hits_at_k[rank_type] = list()
            for k in [1, 3, 10]:
                hits_at_k[rank_type].append(float(torch.mean((ranks <= k).float())))

            ranks = ranks.tolist()
            rankings[rank_type] = ranks

    return (mrr, hits_at_k, rankings, [node_embeddings, edge_embeddings])


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
    num_relations = int(A.shape[1]/num_nodes)

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

    if len(indices) <= 0:
        return

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


def compute_ranks_fast(data, node_embeddings, edge_embeddings,
                       batch_size=16, filtered=True, term_width=80):
    true_heads, true_tails = truedicts(data) if filtered else (None, None)

    num_facts = data.shape[0]
    num_nodes = node_embeddings.shape[0]
    num_batches = int((num_facts + batch_size-1)//batch_size)
    ranks = torch.empty((num_facts*2), dtype=torch.int64)
    for head in [False, True]:  # head or tail prediction
        offset = int(head) * num_facts
        for batch_id in range(num_batches):
            batch_begin = batch_id * batch_size
            batch_end = min(num_facts, (batch_id+1) * batch_size)

            batch_idx = (int(head) * num_batches) + batch_id + 1
            if batch_idx % min(ceil(2*num_batches/20), 100) == 0:
                batch_str = " [DistMult] - batch %2.d / %d" % (batch_idx,
                                                               num_batches*2)
                width_remain = term_width - len(batch_str)
                batch_str += " " * (width_remain//2)  # ensure overwrite of batch message
                print(batch_str, end='\b'*len(batch_str), flush=True)

            batch_data = data[batch_begin:batch_end]
            batch_num_facts = batch_data.shape[0]

            # compute the full score matrix (filter later)
            bases = batch_data[:, 1:] if head else batch_data[:, :2]
            targets = batch_data[:, 0] if head else batch_data[:, 2]

            # collect the triples for which to compute scores
            bexp = bases.view(batch_num_facts, 1, 2).expand(batch_num_facts,
                                                            num_nodes, 2)
            ar   = torch.arange(num_nodes).view(1, num_nodes, 1).expand(batch_num_facts,
                                                                        num_nodes, 1)
            candidates = torch.cat([ar, bexp] if head else [bexp, ar], dim=2)

            scores = score_distmult_bc((candidates[:, :, 0],
                                        candidates[:, :, 1],
                                        candidates[:, :, 2]),
                                       node_embeddings,
                                       edge_embeddings).to('cpu')

            # filter out the true triples that aren't the target
            if filtered:
                filter_scores_(scores, batch_data, true_heads, true_tails, head=head)

            # Select the true scores, and count the number of values larger than than
            true_scores = scores[torch.arange(batch_num_facts), targets]
            batch_ranks = torch.sum(scores > true_scores.view(batch_num_facts, 1), dim=1, dtype=torch.int64)
            # -- This is the "optimistic" rank (assuming it's sorted to the front of the ties)
            num_ties = torch.sum(scores == true_scores.view(batch_num_facts, 1), dim=1, dtype=torch.int64)

            # Account for ties (put the true example halfway down the ties)
            batch_ranks = batch_ranks + torch.round((num_ties - 1) / 2).long()

            ranks[offset+batch_begin:offset+batch_end] = batch_ranks

    return ranks + 1


def score_distmult_bc(data, node_embeddings, edge_embeddings):
    si, pi, oi = data

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
