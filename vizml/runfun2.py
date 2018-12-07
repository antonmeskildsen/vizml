import operator
import networkx as nx
from nxpd import draw

from graph import Session
from nodes import Variable, Input, Output, Constant



X = Input('X')
W1 = Variable(1, 'W1')

a1 = X*W1

Y = Input('Y')
W2 = Variable(2, 'W2')

a2 = Y*W2

a3 = X * a2

out = Output(a1 + a2 + a3)




ctx = {
    'X': 2,
    'Y': 3
}

for node in out.graph():
    print(node)
    if isinstance(node, Input):
        x = node.compute(ctx[node.name])
    elif isinstance(node, Variable) or isinstance(node, Constant):
        x = node.compute()
    else:
        inputs = [n.value for n in node.input_nodes]
        x = node.compute(*inputs)

print(x)