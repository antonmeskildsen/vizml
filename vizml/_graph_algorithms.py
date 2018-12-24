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


def traverse_post_order(operation):
    """Performs a post-order traversal, returning a list of nodes
    in the order in which they have to be computed

    Args:
       operation: The operation to start traversal at
    """

    nodes_post_order = []

    def recurse(node):
        for input_node in node.input_nodes:
            recurse(input_node)
        nodes_post_order.append(node)

    recurse(operation)
    return nodes_post_order
