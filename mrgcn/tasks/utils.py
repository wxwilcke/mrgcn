#!/usr/bin/env python

import logging

import numpy as np
from rdflib.term import URIRef


logger = logging.getLogger(__name__)

def strip_graph(knowledge_graph, config):
    target_property_inv = config['task']['target_property_inv']
    if target_property_inv == '':
        return

    n = len(knowledge_graph)
    separate_literals = config['graph']['structural']['separate_literals']
    logger.debug("Stripping knowledge graph...")
    # remove inverse target relations to prevent information leakage
    inv_target_triples = frozenset(knowledge_graph.triples((None,
                                                      URIRef(target_property_inv),
                                                      None),
                                                    separate_literals))
    knowledge_graph.graph -= inv_target_triples

    m = len(knowledge_graph)
    logger.debug("stripped {} triples ({} remain)".format(n-m, m))

def dataset_to_device(dataset, device):
    for split in dataset.values():
        split['Y'] = split['Y'].to(device)
        split['idx'] = split['idx'].to(device)
        # X stays where it is

def mkbatches(mat, node_idx, num_batches=1):
    """ split N x * array in batches
    """
    n = mat.shape[0]  # number of samples
    num_batches = min(n, num_batches)
    idc = np.arange(n, dtype=np.int32)

    if num_batches <= 1:
        logger.debug("Full batch mode")

    idc_assignments = np.array_split(idc, num_batches)
    node_assignments = [np.array(node_idx, dtype=np.int32)[slce]
                        for slce in idc_assignments]

    return list(zip(idc_assignments, node_assignments))

def mkbatches_varlength(sequences, node_idx, seq_length_map,
                        num_batches=1):
    n = len(sequences)
    num_batches = min(n, num_batches)
    if num_batches <= 1:
        logger.debug("Full batch mode")

    # sort on length
    idc = np.arange(n, dtype=np.int32)
    _, sequences_sorted_idc = zip(*sorted(zip(seq_length_map, idc)))

    seq_assignments = np.array_split(sequences_sorted_idc, num_batches)
    node_assignments = [np.array(node_idx, dtype=np.int32)[slce]
                        for slce in seq_assignments]

    return list(zip(seq_assignments, node_assignments))

def trim_outliers(sequences, node_idx, seq_length_map, feature_dim=0):
    # split outliers
    q25 = np.quantile(seq_length_map, 0.25)
    q75 = np.quantile(seq_length_map, 0.75)
    IQR = q75 - q25
    cut_off = IQR * 1.5

    if IQR <= 0.0:  # no length difference
        return [sequences, node_idx, seq_length_map]

    sequences_trimmed = list()
    seq_length_map_trimmed = list()
    for i, seq_length in enumerate(seq_length_map):
        sequence = sequences[i]
        threshold = int(q75 + cut_off)
        if seq_length > threshold:
            sequence = sequence.tolil()[:, :threshold].tocoo() if feature_dim == 0\
                else sequence.tolil()[:threshold, :].tocoo()

        sequences_trimmed.append(sequence)
        seq_length_map_trimmed.append(sequence.shape[1-feature_dim])

    n = len(sequences_trimmed)
    d = len(sequences) - n
    if d > 0:
        logger.debug("Trimmed {} outliers)".format(d))

    return [sequences_trimmed, node_idx, seq_length_map_trimmed]

def remove_outliers(sequences, node_idx, seq_length_map):
    # split outliers
    q25 = np.quantile(seq_length_map, 0.25)
    q75 = np.quantile(seq_length_map, 0.75)
    IQR = q75 - q25
    cut_off = IQR * 1.5

    if IQR <= 0.0:  # no length difference
        return [sequences, node_idx, seq_length_map]

    sequences_filtered = list()
    node_idx_filtered = list()
    seq_length_map_filtered = list()
    for i, seq_length in enumerate(seq_length_map):
        if seq_length < q25 - cut_off or seq_length > q75 + cut_off:
            # skip outlier
            continue

        sequences_filtered.append(sequences[i])
        node_idx_filtered.append(node_idx[i])
        seq_length_map_filtered.append(seq_length)

    n = len(sequences_filtered)
    d = len(sequences) - n
    if d > 0:
        logger.debug("Filtered {} outliers ({} remain)".format(d, n))

    return [sequences_filtered, node_idx_filtered, seq_length_map_filtered]

def triples_to_indices(kg, node_map, edge_map, separate_literals=False):
    data = np.zeros((len(kg), 3), dtype=np.int32)
    for i, (s, p, o) in enumerate(kg.triples(separate_literals=separate_literals)):
        data[i] = np.array([node_map[s], edge_map[p], node_map[o]], dtype=np.int32)

    return data
