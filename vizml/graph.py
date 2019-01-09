import os
from collections import defaultdict

import numpy as np
import pydotplus as pd

from _graph_algorithms import DepthFirstOrder

# TODO Fix mutual imports
from nodes import Variable, Constant, Input, Output
from operations import Operation


def of(*nodes):
    g = Graph()
    for node in nodes:
        g.add_all(node)
    return g


def _type_to_key(node):
    for typ, key in Graph.type_conv:
        if issubclass(node.__class__, typ):
            return key
    return None


class Graph:

    def __init__(self):
        self.nodes = []
        self.show_values = False
        self.show_gradients = False

    default_attrs = {'style': 'filled', 'fontname': 'Helvetica'}

    attr_map = defaultdict(lambda: Graph.default_attrs, {
        'variable': {'fillcolor': '#27AE60', **default_attrs},
        'operation': {'fillcolor': '#D35400', **default_attrs},
        'constant': {'fillcolor': '#3498DB', **default_attrs},
        'input': {'fillcolor': '#3498DB', **default_attrs},
        'output': {'fillcolor': '#1ABC9C', **default_attrs},
    })

    type_conv = [
        (Input, 'input'),
        (Output, 'output'),
        (Constant, 'constant'),
        (Variable, 'variable'),
        (Operation, 'operation'),
    ]

    def to_dot(self, dot):
        m = {node: i for i, node in enumerate(self.topological())}

        for node in self:
            attrs = Graph.attr_map[_type_to_key(node)]
            dot.add_node(pd.Node(m[node], label=f'"{node}"', **attrs))
            
            if not self.show_gradients or self.show_values:
                label = f'{node.value}' if node.value is not None else ''
                for c in node.consumers:
                    if c in self.nodes:
                        dot.add_edge(pd.Edge(m[node], m[c], label=label))
            
            if self.show_gradients and node.gradient_values is not None:
                for i, input_node in enumerate(node.input_nodes):
                    if len(node.input_nodes) > 1:
                        val = node.gradient_values[i]
                    else:
                        val = node.gradient_values
                    dot.add_edge(pd.Edge(m[node],
                                         m[input_node],
                                         label=str(val)))

    def export(self, open_in_editor=False):
        out = pd.Dot()
        self.to_dot(out)
        out.set_graph_defaults(fontname='"helvetica"')
        out.write('resources/t4.pdf', format='pdf')
        if open_in_editor:
            os.system('open /Users/Anton/Documents/git/vizml/resources/t4.pdf')

    def draw(self):
        out = pd.Dot()
        self.to_dot(out)
        out.set_graph_defaults(fontname='"helvetica"')
        return out.create(format='svg').decode('utf-8')

    def _repr_html_(self):
        return self.draw()

    def add_all(self, node):
        if node is None:
            return
        
        self.nodes.append(node)

        for n in node.inputs():
            if n not in self.nodes:
                self.add_all(n)
    
    def topological(self):
        reverse_post = DepthFirstOrder(self).reverse_post
        return reverse_post

    def topological_reverse(self):
        return reversed(self.topological())
        
    def __iter__(self):
        return iter(self.topological())
