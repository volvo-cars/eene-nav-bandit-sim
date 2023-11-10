# -*- coding: utf-8 -*-

from collections import MutableMapping


class Edge:
    def __init__(self, length):
        self.length = length


class VertexOutgoingEdges(MutableMapping):

    def __init__(self, road_graph):
        self.road_graph = road_graph
        self.edges = dict()

    def __setitem__(self, key, val):
        if self.edges[key] is None:
            self.edges[key] = Edge()
        self.edges[key].weight = val

    def __getitem__(self, key):
        e = self.edges[key]
        return self.road_graph._edge_weight_fun(e)

    def __delitem__(self, key):
        del self.edges[key]

    def __iter__(self):
        return iter(self.edges)

    def __len__(self):
        return len(self.edges)

    def __str__(self):
        return str(self.edges)

    def __repr__(self):
        return '{}, VertexOutgoingEdges({})'.format( \
            super(VertexOutgoingEdges, self).__repr__(), self.edges)

    def get_edge(self, key):
        return self.edges[key]

    def set_edge(self, key, value):
        self.edges[key] = value


class RoadGraph(MutableMapping):

    def __init__(self):
        self.graph = dict()
        self.set_edge_weight_function()

    def __setitem__(self, key, val):
        self.graph[key] = val

    def __getitem__(self, key):
        return self.graph[key]

    def __delitem__(self, key):
        del self.graph[key]

    def __iter__(self):
        return iter(self.graph)

    def __len__(self):
        return len(self.graph)

    def __repr__(self):
        return '{}, RoadGraph({})'.format( \
            super(RoadGraph, self).__repr__(), self.graph)

    def set_edge_weight_function(self, fun=None):
        if fun is None or fun == 'length':
            self._edge_weight_fun = lambda e: e.weight if hasattr(e, 'weight') else e.length
        elif fun == 'time':
            self._edge_weight_fun = lambda e: e.weight if hasattr(e, 'weight') else e.time
        elif callable(fun):
            self._edge_weight_fun = fun
        else:
            raise ValueError("Edge weight function must be " +
                             "'length', 'time' or a function!")

    def get_edge_weight_function(self):
        return self._edge_weight_fun

    def reverse(self):
        reverse_graph = RoadGraph()
        for from_edge in self.graph:
            for to_edge in self.graph[from_edge]:
                if to_edge not in reverse_graph:
                    reverse_graph[to_edge] = VertexOutgoingEdges(reverse_graph)
                reverse_graph[to_edge].set_edge(from_edge, self.graph[from_edge].get_edge(to_edge))
        return reverse_graph
