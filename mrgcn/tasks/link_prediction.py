#!/usr/bin/python3

import logging
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mrgcn.data.io.knowledge_graph import KnowledgeGraph
from mrgcn.encodings.graph_features import construct_features
from mrgcn.models.mrgcn import MRGCN


logger = logging.getLogger(__name__)

def run(A, X, C, nodes_map, edges_map, tsv_writer, device, config,
        modules_config, featureless):
    tsv_writer.writerow(["epoch", "training_loss", "training_accurary",
                                  "validation_loss", "validation_accuracy",
                                  "test_loss", "test_accuracy"])

    # load triples
    data = dict()
    for split in ("train", "valid", "test"):
        with KnowledgeGraph(graph=config['graph'][split]) as kg_split:
            data[split] = set(kg_split.graph)

    # compile model
    model = build_model(C, A, modules_config, config, featureless)
    optimizer = optim.Adam(model.parameters(),
                           lr=config['model']['learning_rate'],
                           weight_decay=config['model']['l2norm'])
    criterion = nn.BCEWithLogitsLoss()

    # mini batching
    mini_batch = config['model']['mini_batch']

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
    for epoch in train_model(A, X, data, nodes_map, edges_map, model, optimizer,
                             criterion, nepoch, mini_batch, device):
        # log metrics
        tsv_writer.writerow([str(epoch[0]),
                             str(epoch[1]),
                             str(epoch[2]),
                             str(epoch[3]),
                             str(epoch[4]),
                             "-1", "-1"])

        # early stopping
        val_loss = epoch[3]
        if patience <= 0:
            continue
        if best_score < 0:
            best_score = val_loss
            best_state = model.state_dict()
        if val_loss >= best_score - delta:
            patience_left -= 1
        else:
            best_score = val_loss
            best_state = model.state_dict()
            patience_left = patience
        if patience_left <= 0:
            model.load_state_dict(best_state)
            logger.info("Early stopping after no improvement for {} epoch".format(patience))
            break

    logging.info("Training time: {:.2f}s".format(time()-t0))

    # test model
    test_loss, test_acc = test_model(A, X, data, nodes_map, edges_map, model,
                                     criterion, device)
    # log metrics
    tsv_writer.writerow(["-1", "-1", "-1", "-1", "-1",
                         str(test_loss), str(test_acc)])

    return (test_loss, test_acc)

def train_model(A, X, data, nodes_map, edges_map, model, optimizer, criterion,
                nepoch, mini_batch, device):
    logging.info("Training for {} epoch".format(nepoch))
    model.train(True)
    for epoch in range(1, nepoch+1):
        batch_grad_idx = epoch - 1
        if not mini_batch:
            batch_grad_idx = -1

        # Single training iteration
        node_embeddings = model(X, A,
                                batch_grad_idx=batch_grad_idx,
                                device=device)
        edge_embeddings = model.rcgn.relations


        # DistMult
        scores = dict()
        for split in ('train', 'valid'):
            if split == "train":
                model.train()
            else:
                model.eval()

            # sample negative triples
            nsamples = len(data[split])
            neg_samples_idx = np.random.choice(np.arange(nsamples),
                                               nsamples//2,
                                               replace=False)

            # embed and score triples
            n = len(data[split])
            s_indices = np.zeros(n)
            p_indices = np.zeros(n)
            o_indices = np.zeros(n)
            for i, (s, p, o) in enumerate(data[split]):
                s_idx = nodes_map[s]
                p_idx = edges_map[p]
                o_idx = nodes_map[o]
                if i in neg_samples_idx:
                    corrupt_idx = np.random.choice(np.arange(nsamples))
                    if np.random.rand() > 0.5:
                        # corrupt head
                        s_idx = corrupt_idx
                    else:
                        # corrupt tail
                        o_idx = corrupt_idx

                s_indices[i] = s_idx
                p_indices[i] = p_idx
                o_indices[i] = o_idx

            Y_hat = score_distmult(node_embeddings[s_indices],
                                   edge_embeddings[p_indices],
                                   node_embeddings[o_indices])

            # create labels; positive samples are 1, negative 1
            Y = np.ones(n, dtype=np.int8)
            Y[neg_samples_idx] = 0

            # scores
            scores[split] = (binary_crossentropy(Y_hat, Y, criterion),
                             compute_accuracy(Y_hat, Y))

            if split == "train":
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                scores[split][0].backward()  # training loss
                optimizer.step()

        # DEBUG #
        #for name, param in model.named_parameters():
        #    logger.info(name + " - grad mean: " + str(float(param.grad.mean())))
        # DEBUG #

        # cast criterion objects to floats to free the memory of the tensors
        # they point to
        train_loss = float(scores["train"][0])
        train_acc = float(scores["train"][1])
        val_loss = float(scores["valid"][0])
        val_acc = float(scores["valid"][1])
        del scores

        logging.info("{:04d} ".format(epoch) \
                     + "| train loss {:.4f} / acc {:.4f} ".format(train_loss,
                                                                  train_acc)
                     + "| val loss {:.4f} / acc {:.4f}".format(val_loss,
                                                               val_acc))

        yield (epoch,
               train_loss, train_acc,
               val_loss, val_acc)

def test_model(A, X, data, nodes_map, edges_map, model, criterion, device):
    # Predict on full dataset
    model.train(False)
    with torch.no_grad():
        node_embeddings = model(X, A,
                                batch_grad_idx=-1,
                                device=device)
        edge_embeddings = model.rcgn.relations

    # DistMult
    # sample negative triples
    nsamples = len(data["test"])
    neg_samples_idx = np.random.choice(np.arange(nsamples),
                                       nsamples//2,
                                       replace=False)

    # embed and score triples
    n = len(data["test"])
    s_indices = np.zeros(n)
    p_indices = np.zeros(n)
    o_indices = np.zeros(n)
    for i, (s, p, o) in enumerate(data["test"]):
        s_idx = nodes_map[s]
        p_idx = edges_map[p]
        o_idx = nodes_map[o]
        if i in neg_samples_idx:
            corrupt_idx = np.random.choice(np.arange(nsamples))
            if np.random.rand() > 0.5:
                # corrupt head
                s_idx = corrupt_idx
            else:
                # corrupt tail
                o_idx = corrupt_idx

        s_indices[i] = s_idx
        p_indices[i] = p_idx
        o_indices[i] = o_idx

    Y_hat = score_distmult(node_embeddings[s_indices],
                           edge_embeddings[p_indices],
                           node_embeddings[o_indices])

    # create labels; positive samples are 1, negative 1
    Y = np.ones(n, dtype=np.int8)
    Y[neg_samples_idx] = 0

    # scores on test set
    test_loss = binary_crossentropy(Y_hat, Y, criterion)
    test_acc = compute_accuracy(Y_hat, Y)

    test_loss = float(test_loss)
    test_acc = float(test_acc)

    logging.info("Performance on test set: loss {:.4f} / accuracy {:.4f}".format(
                  test_loss,
                  test_acc))

def build_dataset(knowledge_graph, nodes_map, config, featureless):
    logger.debug("Starting dataset build")
    if featureless:
        F = dict()
    else:
        separate_literals = config['graph']['structural']['separate_literals']
        F = construct_features(nodes_map, knowledge_graph,
                               config['graph']['features'],
                               separate_literals)

    Y = torch.empty((0, 0), dtype=torch.float16)  # dummy

    logger.debug("Completed dataset build")

    return (F, Y)

def build_model(C, A, modules_config, config, featureless):
    layers = config['model']['layers']
    assert len(layers) >= 2
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

def compute_accuracy(Y_hat, Y):
    Y = torch.as_tensor(Y, dtype=torch.long)
    return torch.mean(torch.eq(Y_hat, Y).float())

def score_distmult(self, s, p, o):
    return torch.sum(s * p * o, dim=1)
