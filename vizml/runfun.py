import operator
import networkx as nx
from nxpd import draw

from graph import Graph, Session, _default_graph
from nodes import *

g = Graph()
g.as_default()


A = Variable(1, 'A')
B = Variable(2, 'B')

x = Placeholder('x')

z = B * (A + (x**2))
res = Output(z, 'result')

#z = A*x

#d = Graph()
#d.as_default()

#back(z)


session = Session()
session.run(g, {
    x: 3
})

print(z.output)

ng = g.to_nx()

# import matplotlib.pyplot as plt

# plt.figure()
# nx.draw(ng, with_labels=True)
# plt.show()

from nxpd import draw


draw(ng, args=['-Nfontname=Fira Code Regular'], format='pdf')