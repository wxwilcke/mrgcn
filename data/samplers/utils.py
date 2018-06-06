#!/usr/bin/python3

from collections import deque
import logging


logger = logging.getLogger(__name__)

def greedy_depth_first_sampler(knowledge_graph=None, atom=None, depth=1, allow_loops=False, _visited=None):
    """ recursive greedy depth-first approach to built up graph around atom
    :param knowledge_graph: a KnowledgeGraph instance to sample
    :param atom: the individual to start from
    :param depth: the maximum number of steps from atom to sample
    :param allow_loops: true if loops are allowed
    :returns: a set of facts
    """
    _facts = set()
    if depth <= 0:
        return _facts
    if _visited is None:
        _visited = set()

    _visited.add(atom)
    for s, p, o in knowledge_graph.graph.triples((atom, None, None)):
        if not allow_loops and o in _visited:
            continue

        _facts.add((s, p, o))
        _facts = _facts.union(greedy_depth_first_sampler(knowledge_graph, o, depth-1, allow_loops, _visited))

    return _facts

def depth_first_sampler(knowledge_graph=None, atom=None, depth=1, allow_loops=False, _visited=None):
    """ recursive depth-first approach to built up graph around atom
    param knowledge_graph: a KnowledgeGraph instance to sample
    :param atom: the individual to start from
    :param depth: the maximum number of steps from atom to sample
    :param allow_loops: true if loops are allowed
    :returns: a set of facts
    """
    _facts = set()
    if depth <= 0:
        return _facts
    if _visited is None:
        _visited = set()

    _visited.add(atom)
    for s, p, o in knowledge_graph.graph.triples((atom, None, None)):
        if not allow_loops and o in _visited:
            continue

        _facts = _facts.union(depth_first_sampler(knowledge_graph, o, depth-1, allow_loops, _visited))
        _facts.add((s, p, o))

    return _facts

def breadth_first_sampler(knowledge_graph=None, atom=None, depth=1, allow_loops=False):
    """ breadth-first approach to built up graph around atom
    :param knowledge_graph: a KnowledgeGraph instance to sample
    :param atom: the individual to start from
    :param depth: the maximum number of steps from atom to sample
    :param allow_loops: true if loops are allowed
    :returns: a set of facts
    """
    facts = set()
    q = deque()

    visited = {atom}
    q.append((atom, 0))
    while len(q) > 0:
        vertex, level = q.popleft()
        if depth == 0 or level >= depth:
            continue

        for s, p, o in knowledge_graph.graph.triples((vertex, None, None)):
            if (s, p, o) in facts:
                continue
            if not allow_loops and o in visited:
                continue

            facts.add((s, p, o))
            q.append((o, level+1))
            visited.add(o)

    return facts
