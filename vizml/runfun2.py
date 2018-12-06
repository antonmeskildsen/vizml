import operator
import networkx as nx
from nxpd import draw

from graph import Session
from nodes import Variable, Input, Output, Constant


A = Variable(1, 'A')
B = Variable(2, 'B')

x = Input('x')

z = B * (A + ((x+A)**2))
res = Output(z, 'result')


z._repr_html_()

from gvanim import Animation
from gvanim.jupyter import interactive
ga = Animation()
