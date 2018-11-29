import numpy as np

from nodes import *

_default_graph = None

class Graph:

    def __init__(self):
        self.nodes = []
    
    def as_default(self):
        global _default_graph
        _default_graph = self
    
    def to_nx(self):
        reverse_post = DepthFirstOrder(self).reverse_post

        g = nx.DiGraph()

        m = {node: i for i, node in enumerate(self.nodes)}

        for node in reverse_post:
            g.add_node(m[node], label=f'"{node}"')
            for c in node.consumers:
                    g.add_edge(m[node], m[c])
        
        return g
                    

class DepthFirstOrder:

    def __init__(self, graph):
        self.marked = {node: False for node in graph.nodes}

        self.reverse_post = []

        for node in graph.nodes:
            if not self.marked[node]:
                self.dfs(node)
    
    def dfs(self, node):
        self.marked[node] = True
        if type(node) == Operation:
            for li in node.input_nodes:
                for w in li:
                    if not self.marked[w]:
                        self.dfs(w)

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

        for node in nodes_postorder:

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