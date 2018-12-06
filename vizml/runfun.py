import operator
import networkx as nx
from nxpd import draw

from graph import Graph, Session
from nodes import Variable, Input, Output


A = Variable(1, 'A')
B = Variable(2, 'B')

x = Input('x')

z = B * (A + (x**2))
res = Output(z, 'result')

#z = A*x

#d = Graph()
#d.as_default()

#back(z)

#### NOTE: Use iterators / generators for calculation (more instructive than magic)

g = Graph.from_node(res)


session = Session()
session.run(g, {
    x: 3
})

print(z.output)

print(type(Variable))

g.draw()