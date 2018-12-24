from typing import List
from abc import ABC, abstractmethod

import numpy as np
from vizml.nodes import *


class Operation(Node, ABC):

    @abstractmethod
    def symbol(self):
        ...

    def __init__(self, input_nodes: List[Node] = None):
        super().__init__(input_nodes)
        for node in input_nodes:
            node.consumers.append(self)

    def __str__(self):
        return self.symbol


class Binary(Operation, ABC):

    def __init__(self, l, r):
        super().__init__([l, r])

    def forward(self, *inputs, **placeholders):
        self.forward_op(inputs[0], inputs[1])

    @abstractmethod
    def forward_op(self, l, r):
        ...


class Add(Binary):
    symbol = '+'

    def forward_op(self, l, r):
        return np.add(l, r)

    def backward(self, gradient, ctx):
        return gradient, gradient


class Sub(Binary):
    symbol = '-'

    def forward_op(self, l, r):
        return np.subtract(l, r)

    def backward(self, gradient, ctx):
        ...


class Mul(Binary):
    symbol = '*'

    def forward_op(self, l, r):
        return np.multiply(l, r)

    def backward(self, gradient, ctx):
        return gradient * ctx[self][0], gradient * ctx[self][1]


class Div(Binary):
    symbol = '/'

    def forward_op(self, l, r):
        return np.divide(l, r)

    def backward(self, gradient, ctx):
        ...


class Pow(Binary):
    symbol = '^'

    def forward_op(self, l, r):
        return np.power(l, r)

    def backward(self, gradient, ctx):
        base_node, exponent_node = self.inputs
        base, exponent = ctx[base_node], ctx[exponent_node]
        return (gradient * exponent * np.power(base, exponent - 1),
                gradient * ctx[self] * np.log(base))
