import os
from collections import defaultdict

import numpy as np
import pydotplus as pd

from nodes import Variable, Operation, Constant, Input, Output


def of(*nodes):
    g = Graph()
    for node in nodes:
        g.add_all(node)
    return g

class Graph:

    def __init__(self):
        self.nodes = []
        self.show_values = False
        self.show_gradients = False

    default_attrs = { 'style': 'filled', 'fontname': 'Helvetica'}

    attr_map = defaultdict(lambda: Graph.default_attrs, {
        'variable': {'fillcolor': '#27AE60', **default_attrs},
        'operation': { 'fillcolor': '#D35400', **default_attrs},
        'constant': { 'fillcolor': '#3498DB', **default_attrs},
        'input': { 'fillcolor': '#3498DB', **default_attrs},
        'output': { 'fillcolor': '#1ABC9C', **default_attrs},
    })

    type_conv = [
        (Input, 'input'),
        (Output, 'output'),
        (Constant, 'constant'),
        (Variable, 'variable'),
        (Operation, 'operation'),
    ]

    def type_to_key(self, node):
        for typ, key in Graph.type_conv:
            if issubclass(node.__class__, typ):
                return key
        return None

    def to_dot(self, dot):
        m = {node: i for i, node in enumerate(self.topological())}

        for node in self:
            attrs = Graph.attr_map[self.type_to_key(node)]
            dot.add_node(pd.Node(m[node], label=f'"{node}"', **attrs))
            
            if not self.show_gradients:
                l = f'{node.value}' if node.value is not None and self.show_values else ''
                for c in node.consumers:
                    if c in self.nodes:
                        dot.add_edge(pd.Edge(m[node], m[c], label=l))
            
            if self.show_gradients and node.gradient_values is not None:
                for i, input_node in enumerate(node.input_nodes):
                    val = node.gradient_values[i] if len(node.input_nodes) > 1 else node.gradient_values
                    dot.add_edge(pd.Edge(m[node], m[input_node], label=str(val)))


    def export(self, open_in_editor=False):
        out = pd.Dot()
        self.to_dot(out)
        out.set_graph_defaults(fontname='"helvetica"')
        out.write('resources/t4.pdf', format='pdf')
        if open_in_editor:
            os.system('open /Users/Anton/Documents/git/vizml/resources/t4.pdf')
        #ng = self.to_nx()
        #draw(ng, args=['-Nfontname=Fira Code Regular'], format='dot', filename='resources/t2.dot', prefix='hej')
    
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
    

class DepthFirstOrder:

    def __init__(self, graph):
        self.marked = {node: False for node in graph.nodes}

        self.reverse_post = []

        for node in graph.nodes:
            if not self.marked[node]:
                self.dfs(node)
    
    def dfs(self, node):
        self.marked[node] = True
        for n in node.input_nodes:
            if not self.marked[n]:
                self.dfs(n)

        self.reverse_post.append(node)


def back(n, a=None):
    if a is None:
        a = Constant(1)
    if isinstance(n, Operation):
        ol = n.backward(a)
        for i, o in zip(n.input_nodes, ol):
            back(i, o)


def traverse_postorder(operation):
    """Performs a post-order traversal, returning a list of nodes
    in the order in which they have to be computed

    Args:
       operation: The operation to start traversal at
    """

    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder


class Session:

    def run(self, graph, ctx):
        for node in graph.graph():
            if isinstance(node, Input):
                if node.name not in ctx:
                    raise ValueError(f'Value for Input <{node.name}> was not provided in context')
                x = node.compute(ctx[node.name])
            elif issubclass(node.__class__, Variable):
                x = node.compute()
            else:
                inputs = [n.value for n in node.input_nodes]
                x = node.compute(*inputs)
        return x