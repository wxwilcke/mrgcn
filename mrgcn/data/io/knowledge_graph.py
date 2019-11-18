#!/usr/bin/python3

from collections import Counter
import logging
import gzip

from rdflib.graph import Graph
from rdflib.term import BNode, Literal
from rdflib.util import guess_format

from mrgcn.data.utils import is_gzip, is_readable


class KnowledgeGraph:
    """ Knowledge Graph Class
    A wrapper around an imported rdflib.Graph object with convenience functions
    """
    graph = None
    _property_distribution = {}

    def __init__(self, graph=None):
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initiating Knowledge Graph")

        if graph is not None:
            if type(graph) is Graph:
                self.graph = graph
            elif type(graph) is str:
                self.graph = self._read(graph)
            else:
                raise TypeError(":: Wrong input type: {}; requires path to RDF"
                                " graph or rdflib.graph.Graph object".format(type(graph)))
        else:
            self.graph = Graph()

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
        seen = set()
        for s, p, o in self.graph.triples((None, None, None)):
            for atom in (s, o):
                if isinstance(atom, Literal):
                    atom = self.UniqueLiteral(s, p, atom)
                if atom in seen:
                    continue
                seen.add(atom)

                yield atom

    def non_terminal_atoms(self):
        self.logger.debug("Yielding non-terminal atoms")
        for atom in frozenset(self.graph.subjects()):
            yield(atom)

    def terminal_atoms(self):
        self.logger.debug("Yielding terminal atoms")
        non_terminal_atoms = list(self.non_terminal_atoms())
        for atom in list(self.graph.objects()):
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
        # return unique properties
        attributes = frozenset(self.attributes())
        self.logger.debug("Yielding OT predicates")
        for p in self.graph.predicates():
            if len(set(self.graph.objects(None, p))-attributes) <= 0:
                # p is only used with a literal as object
                continue

            yield(p)

    def datatype_properties(self):
        # return unique properties
        objecttype_properties = set(self.objecttype_properties())
        self.logger.debug("Yielding DT predicates")
        for p in self.graph.predicates():
            if p in objecttype_properties:
                continue

            yield(p)

    def properties(self):
        self.logger.debug("Yielding properties")
        for p in self.graph.predicates():
            yield(p)

    def triples(self, triple=(None, None, None)):
        self.logger.debug("Yielding triples (triple {})".format(triple))
        for s,p,o in self.graph.triples(triple):
            if isinstance(o, Literal):
                o = self.UniqueLiteral(s, p, o)
            yield s, p, o

    ## Statistics
    def property_frequency(self, property=None):
        if property is None:
            return self._property_distribution
        elif property in self._property_distribution:
            return self._property_distribution[property]

    def attribute_frequency(self, property, limit=None):
        attribute_freq = Counter(self.graph.objects(None, property))
        if limit is None:
            return attribute_freq.most_common()
        else:
            return attribute_freq.most_common(limit)

    ## Operators
    def sample(self, strategy=None, **kwargs):
        """ Sample this graph using the given strategy
        returns a KnowledgeGraph instance
        """
        if strategy is None:
            raise ValueError('Strategy cannot be left undefined')

        self.logger.debug("Sampling graph")
        return strategy.sample(self, **kwargs)

    class UniqueLiteral(Literal):
        # literal with unique hash, irrespective of content
        def __new__(cls, s, p, o):
            self = super().__new__(cls, str(o), o.language, o.datatype, normalize=None)
            self.s = str(s)
            self.p = str(p)

            return self

        def __hash__(self):
            base = self.s + self.p + str(self)
            for attr in [self.language, self.datatype]:
                if attr is not None:
                    base += str(attr)

            return hash(base)

if __name__ == "__main__":
    print("Knowledge Graph")
