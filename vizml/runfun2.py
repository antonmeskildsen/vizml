import operator
import networkx as nx
from nxpd import draw

import numpy as np

from graph import Session
from nodes import Variable, Input, Output, Constant

# X = Input('X')
# W1 = Variable(2, 'W1')

# a1 = X*W1

# Y = Input('Y')
# W2 = Variable(1, 'W2')

# a2 = Y*W2

# a3 = X * a2

# out = Output(a1 + a2 + a3)

X = Input('X')
V = Variable(2, 'V')

out = X*V



ctx = {
    'X': 3
}

for node in out.graph():
    if isinstance(node, Input):
        x = node.compute(ctx[node.name])
    elif isinstance(node, Variable) or isinstance(node, Constant):
        x = node.compute()
    else:
        inputs = [n.value for n in node.input_nodes]
        x = node.compute(*inputs)


from queue import Queue

q = Queue()

grad_table = {out: 1}

q.put(out)

while not q.empty():
    node = q.get()

    if node != out:
        grad_table[node] = 0
        #fun stuff
        for c in node.consumers:
            c_grad = grad_table[c]
            grad = c.backward(c_grad)

            if len(c.input_nodes) == 1:
                grad_table[node] += grad
            else:
                idx = c.input_nodes.index(node)
                grad_table[node] += grad[idx]
    
    if type(node) == Variable:
        print(node, ' d:', grad_table[node])
    
    for i in node.input_nodes:
        q.put(i)
