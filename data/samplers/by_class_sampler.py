#!/usr/bin/python3

import logging

from rdflib.namespace import RDF
from rdf.term import URIRef

from data.io.knowledge_graph import KnowledgeGraph
from data.samplers.utils import breadth_first_sampler


logger = logging.getLogger(__name__)

def sample(knowledge_graph=None,
           instances=[],
           instances_type=None,
           target_property=None,
           target_classes=[],
           depth=1):
    """ Sample subset of knowledge graph based on specified classes
    :param knowledge_graph: a KnowledgeGraph instance to sample
    :param instances: a list of instances to sample around
    :param instances_type: the type of instances to sample
    :param target_property: the property specifying the classes
    :param target_classes: the values to which target_property may link
    :param depth: the maximum number of steps from atom to sample
    :returns: a KnowledgeGraph instance 
    """

    if instances_type is not None:
        target_property = URIRef(target_property)

        # extend instance list with unspecified matches
        g = knowledge_graph.graph
        for instance in g.subjects(RDF.type, URIRef(instances_type)):
            if instance not in instances and\
               g.value(instance, target_property).toPython() in target_classes:
                instances.append(instance)

    logger.debug("Sampling around {} instance vertices".format(len(instances)))

    # construct subset around instances
    subset = KnowledgeGraph()
    for instance in instances:
        for fact in breadth_first_sampler(knowledge_graph, instance, depth):
            subset.graph.add(fact)

    logger.debug("Sampled {} facts".format(len(subset.graph)))

    return subset
