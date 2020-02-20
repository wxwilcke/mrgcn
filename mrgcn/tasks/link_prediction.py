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

def run(A, X, C, data, tsv_writer, device, config,
        modules_config, featureless):
    tsv_writer.writerow(["epoch", "training_loss", "training_accurary",
                                  "validation_loss", "validation_accuracy",
                                  "test_loss", "test_accuracy"])

    # compile model
    num_nodes = A.shape[0]
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
    for epoch in train_model(A, X, data, num_nodes, model, optimizer,
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
    test_loss, test_acc = test_model(A, X, data, num_nodes, model,
                                     criterion, device)
    # log metrics
    tsv_writer.writerow(["-1", "-1", "-1", "-1", "-1",
                         str(test_loss), str(test_acc)])

    return (test_loss, test_acc)

def train_model(A, X, data, num_nodes, model, optimizer, criterion,
                nepoch, mini_batch, device):
    logging.info("Training for {} epoch".format(nepoch))
    for epoch in range(1, nepoch+1):
        # MRGCN + DistMult
        scores = dict()
        for split in ('train', 'valid'):
            if split == "train":
                batch_grad_idx = epoch - 1
                if not mini_batch:
                    batch_grad_idx = -1

                model.train()
            else:
                batch_grad_idx = -1
                model.eval()

            # Single iteration
            node_embeddings = model(X, A,
                                    batch_grad_idx=batch_grad_idx,
                                    device=device)
            edge_embeddings = model.rgcn.relations

            # sample negative triples
            nsamples = data[split].shape[0]
            neg_samples_idx = np.random.choice(np.arange(nsamples),
                                               nsamples//2,
                                               replace=False)

            # embed and score triples
            s_indices = np.zeros(nsamples)
            p_indices = np.zeros(nsamples)
            o_indices = np.zeros(nsamples)
            for i in range(nsamples):
                s_idx, p_idx, o_idx = data[split][i]
                if i in neg_samples_idx:
                    corrupt_idx = np.random.choice(np.arange(num_nodes))
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

            # create labels; positive samples are 1, negative 0
            Y = np.ones(nsamples, dtype=np.int8)
            Y[neg_samples_idx] = 0

            # scores
            loss = binary_crossentropy(Y_hat, Y, criterion)
            scores[split] = (float(loss),
                             float(compute_accuracy(Y_hat, Y)))

            if split == "train":
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                scores[split][0].backward()  # training loss
                optimizer.step()

            logging.info("{:04d} ".format(epoch) \
                         + "| train loss {:.4f} / acc {:.4f}".format(scores["train"][0],
                                                                     scores["train"][1])
                         + "| val loss {:.4f} / acc {:.4f}".format(scores["valid"][0],
                                                                   scores["valid"][1]))

            yield (epoch, scores["train"][0], scores["train"][1],
                   scores["valid"][0], scores["valid"][1])

def test_model(A, X, data, num_nodes, model, criterion, device):
    # Predict on full dataset
    model.train(False)
    with torch.no_grad():
        node_embeddings = model(X, A,
                                batch_grad_idx=-1,
                                device=device)
        edge_embeddings = model.rgcn.relations

    # DistMult
    # sample negative triples
    nsamples = data["test"].shape[0]
    neg_samples_idx = np.random.choice(np.arange(nsamples),
                                       nsamples//2,
                                       replace=False)

    # embed and score triples
    s_indices = np.zeros(nsamples)
    p_indices = np.zeros(nsamples)
    o_indices = np.zeros(nsamples)
    for i in range(nsamples):
        s_idx, p_idx, o_idx = data["test"][i]
        if i in neg_samples_idx:
            corrupt_idx = np.random.choice(np.arange(num_nodes))
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

    # create labels; positive samples are 1, negative 0
    Y = np.ones(nsamples, dtype=np.int8)
    Y[neg_samples_idx] = 0

    # scores on test set
    test_loss = binary_crossentropy(Y_hat, Y, criterion)
    test_acc = compute_accuracy(Y_hat, Y)

    test_loss = float(test_loss)
    test_acc = float(test_acc)

    logging.info("Performance on test set: loss {:.4f} / accuracy {:.4f}".format(
                  test_loss,
                  test_acc))

    return (test_loss, test_acc)

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
    Y_hat = torch.round(Y_hat)
    return torch.mean(torch.eq(Y_hat, Y).float())

def score_distmult(self, s, p, o):
    return torch.sum(s * p * o, dim=1)
