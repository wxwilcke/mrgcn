#!/usr/bin/python3

import logging

import numpy as np
import scipy.sparse as sp


logger = logging.getLogger(__name__)

def generate(knowledge_graph, config):
    separate_literals = config['graph']['structural']['separate_literals']
    # create mapping to integers [0, ...]
    properties_dict = {prop: i for i, prop in enumerate(knowledge_graph.properties())}
    nodes_dict = {node: i for i, node in
                  enumerate(knowledge_graph.atoms(separate_literals))}
    num_nodes = len(nodes_dict)

    # generate adjacency matrix for each property
    adj_shape = (num_nodes, num_nodes)
    adjacencies = generate_adjacency_matrices(knowledge_graph,
                                properties_dict,
                                nodes_dict,
                                adj_shape,
                                separate_literals,
                                config['graph']['structural'])

    # add identity matrix (self-relations)
    ident = sp.identity(num_nodes).tocsr()
    ident = normalize_adjacency_matrix(ident)
    adjacencies.append(ident)

    # stack into a n x nR matrix
    return [sp.hstack(adjacencies, format="csr"), nodes_dict]

def generate_adjacency_matrices(knowledge_graph,
                                properties_dict,
                                nodes_dict,
                                adj_shape,
                                separate_literals,
                                config):
    include_inverse = config['include_inverse_properties']
    exclude_properties = config['exclude_properties']

    logger.debug("Generating {} adjacency matrices of size {}".format(
        len([p for p in properties_dict.keys() if p not in exclude_properties]),
                                                        adj_shape))
    adjacencies = []
    for prop, _ in properties_dict.items():
        if prop in exclude_properties:
            continue

        # create array to hold all edges per property
        edges = np.empty((knowledge_graph.property_frequency(prop), 2),
                         dtype=np.int32)

        # populate edge array with corresponding node URIs
        for idx, (s, p, o) in enumerate(knowledge_graph.triples((None,
                                                                 prop,
                                                                 None),
                                                               separate_literals)):
            edges[idx] = np.array([nodes_dict[s], nodes_dict[o]])

        # split subject (row) and object (col) node URIs
        row, col = np.transpose(edges)

        # create adjacency matrix for this property
        data = np.ones(len(row), dtype=np.int8)
        adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.int8)
        adj = normalize_adjacency_matrix(adj)
        adjacencies.append(adj)

        # create adjacency matrix for inverse property
        if include_inverse:
            adj = sp.csr_matrix((data, (col, row)), shape=adj_shape, dtype=np.int8)
            adj = normalize_adjacency_matrix(adj)
            adjacencies.append(adj)

    return adjacencies

def normalize_adjacency_matrix(adj):
    d = np.array(adj.sum(1)).flatten()
    d_inv = 1. / d
    d_inv[np.isinf(d_inv)] = 0.
    D_inv = sp.diags(d_inv)

    return D_inv.dot(adj).tocsr()
