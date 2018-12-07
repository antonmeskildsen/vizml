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

    default_attrs = { 'style': 'filled', 'fontname': 'Avenir'}

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
        m = {node: i for i, node in enumerate(self.nodes)}

        for node in self:
            attrs = Graph.attr_map[self.type_to_key(node)]
            dot.add_node(pd.Node(m[node], label=f'"{node}"', **attrs))
            for c in node.consumers:
                if c in self.nodes:
                    dot.add_edge(pd.Edge(m[node], m[c]))


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

    def run(self, graph, feed_dict={}):
        nodes_postorder = DepthFirstOrder(graph).reverse_post

        for node in reversed(nodes_postorder):

            if type(node) == Input:
                # Set the node value to the placeholder value from feed_dict
                node.output = feed_dict[node]
            elif type(node) == Variable or type(node) == Constant:
                # Set the node value to the variable's value attribute
                node.output = node.value
            else:  # Operation
                # Get the input values for this operation from node_values
                node.inputs = [input_node.output for input_node in node.input_nodes]

                # Compute the output of this operation
                node.output = node.compute(*node.inputs)

            # Convert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)