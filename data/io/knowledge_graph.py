#!/usr/bin/python3

from collections import Counter
import logging
from itertools import chain
import gzip

from rdflib.graph import Graph
from rdflib.term import BNode, Literal
from rdflib.util import guess_format

from data.utils import is_gzip, is_readable


class KnowledgeGraph:
    """ Knowledge Graph Class
    A wrapper around an imported rdflib.Graph object with convenience functions
    """
    graph = None
    _property_distribution = {}

    def __init__(self, graph=None, path=None):
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initiating Knowledge Graph")
        
        if graph is not None:
            self.graph = graph
        elif path is not None:
            self.graph = self._read(path)
        else:
            raise ValueError(":: Constructor missing required argument")

        self._property_distribution = Counter(self.graph.predicates())
        self.logger.debug("Knowledge Graph ({} facts) succesfully imported".format(len(self.graph)))

    def _read(self, path=None):
        assert is_readable(path) 
        graph = Graph()

        if not is_gzip(path):
            graph.parse(path, format=guess_format(path))
        else:
            self.logger.debug("Input recognized as gzip file")
            with gzip.open(path, 'rb') as f:
                graph.parse(f, format=guess_format(path[:-3]))

        return graph
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.graph.destroy("store")
        self.graph.close(True)

    def __len__(self):
        return len(self.graph)

    ### Generators ###

    def atoms(self):
        self.logger.debug("Yielding atoms")
        for atom in frozenset(chain(self.graph.subjects(), self.graph.objects())):
            yield(atom)

    def non_terminal_atoms(self):
        self.logger.debug("Yielding non-terminal atoms")
        for atom in frozenset(self.graph.subjects()):
            yield(atom)

    def terminal_atoms(self):
        self.logger.debug("Yielding terminal atoms")
        non_terminal_atoms = list(self.non_terminal_atoms())
        for atom in frozenset(self.graph.objects()):
            if atom in non_terminal_atoms:
                continue

            yield(atom)

    def attributes(self):
        self.logger.debug("Yielding attributes")
        for obj in self.graph.objects():
            if type(obj) is Literal:
                yield(obj)

    def entities(self, omit_blank_nodes=False):
        self.logger.debug("Yielding entities")
        for res in self.atoms():
            if (type(res) is Literal or
               (omit_blank_nodes and type(res) is BNode)):
                continue

            yield(res)

    def objecttype_properties(self):
        attributes = frozenset(self.attributes())
        self.logger.debug("Yielding OT predicates")
        for p in frozenset(self.graph.predicates()):
            if len(set(self.graph.objects(None, p))-attributes) <= 0:
                # p is only used with a literal as object
                continue

            yield(p)
            
    def datatype_properties(self):
        objecttype_properties = set(self.objecttype_properties())
        self.logger.debug("Yielding DT predicates")
        for p in frozenset(self.graph.predicates()):
            if p in objecttype_properties:
                continue

            yield(p)
    
    def properties(self):
        self.logger.debug("Yielding properties")
        for p in frozenset(self.graph.predicates()):
            yield(p)

    def triples(self, property=None):
        self.logger.debug("Yielding triples (property {})".format(property))
        for s,p,o in self.graph.triples((None, property, None)):
            yield s, p, o

    ## Other
    def property_frequency(self, property=None):
        if property is None:
            return self._property_distribution
        elif property in self._property_distribution:
            return self._property_distribution[property]


if __name__ == "__main__":
    print("Knowledge Graph")
