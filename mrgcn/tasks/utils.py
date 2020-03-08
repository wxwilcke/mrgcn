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

def mkbatches(mat, node_idx, C, _, nsets, nepoch, passes_per_batch=1):
    """ split N x * array in batches
    """
    n = mat.shape[0]  # number of samples
    idc = np.array(range(n), dtype=np.int32)

    nbins = nepoch
    if passes_per_batch > 1:
        nbins = nepoch//passes_per_batch

    idc_assignments = np.array_split(idc, nbins)
    node_assignments = [np.array(node_idx, dtype=np.int32)[slce]
                        for slce in idc_assignments]

    return list(zip(idc_assignments, node_assignments))

def mkbatches_varlength(sequences, node_idx, C, seq_length_map, _,
                            nepoch, passes_per_batch=1):
    n = len(sequences)
    # sort on length
    idc = np.array(range(n), dtype=np.int32)
    seq_length_map_sorted, node_idx_sorted, sequences_sorted_idc = zip(
        *sorted(zip(seq_length_map,
                    node_idx,
                    idc)))

    nbins = nepoch
    if passes_per_batch > 1:
        nbins = nepoch//passes_per_batch

    seq_assignments = np.array_split(sequences_sorted_idc, nbins)
    node_assignments = [np.array(node_idx, dtype=np.int32)[slce]
                        for slce in seq_assignments]

    return list(zip(seq_assignments, node_assignments))

def trim_outliers(sequences, node_idx, C, seq_length_map, nsets, feature_dim=0):
    # split outliers
    q25 = np.quantile(seq_length_map, 0.25)
    q75 = np.quantile(seq_length_map, 0.75)
    IQR = q75 - q25
    cut_off = IQR * 1.5

    if IQR <= 0.0:  # no length difference
        return [sequences, node_idx, C, seq_length_map, nsets]

    sequences_trimmed = list()
    seq_length_map_trimmed = list()
    for i, seq_length in enumerate(seq_length_map):
        sequence = sequences[i]
        if seq_length > q75 + cut_off:
            sequence = sequence[:, :q75+cut_off] if feature_dim == 0\
                else sequence[:q75+cut_off, :]

        sequences_trimmed.append(sequence)
        seq_length_map_trimmed.append(sequence.shape[1-feature_dim])

    n = len(sequences_trimmed)
    d = len(sequences) - n
    if d > 0:
        logger.debug("Trimmed {} outliers)".format(d))

    return [sequences_trimmed, node_idx, C, seq_length_map_trimmed, nsets]

def remove_outliers(sequences, node_idx, C, seq_length_map, nsets):
    # split outliers
    q25 = np.quantile(seq_length_map, 0.25)
    q75 = np.quantile(seq_length_map, 0.75)
    IQR = q75 - q25
    cut_off = IQR * 1.5

    if IQR <= 0.0:  # no length difference
        return [sequences, node_idx, C, seq_length_map, nsets]

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

    return [sequences_filtered, node_idx_filtered, C, seq_length_map_filtered, nsets]

def triples_to_indices(kg, node_map, edge_map, separate_literals=False):
    data = np.zeros((len(kg), 3), dtype=np.int32)
    for i, (s, p, o) in enumerate(kg.triples(separate_literals=separate_literals)):
        data[i] = np.array([node_map[s], edge_map[p], node_map[o]], dtype=np.int32)

    return data
