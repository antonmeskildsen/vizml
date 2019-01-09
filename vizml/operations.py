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


class Unary(Operation, ABC):

    def __init__(self, i):
        super().__init__([i])

    def forward(self, *inputs, **placeholders):
        return self.forward_op(inputs[0])

    @abstractmethod
    def forward_op(self, i):
        ...


class Negative(Unary):
    symbol = 'neg'

    def forward_op(self, i):
        return -i

    def backward(self, gradient, ctx):
        return -gradient


class Log(Unary):
    symbol = 'log'

    def forward_op(self, i):
        return np.log(i)

    def backward(self, gradient, ctx):
        input = self.inputs[0]
        return gradient/ctx[input]


class Sigmoid(Unary):
    symbol = '𝜎'

    def forward_op(self, i):
        ex = np.exp(i)
        return ex/(ex+1)

    def backward(self, gradient, ctx):
        sigmoid = ctx[self]
        return gradient * sigmoid * (1 - sigmoid)


class ReLU(Unary):
    symbol = 'relu'

    def forward_op(self, i):
        return max(0, i)

    def backward(self, gradient, ctx):
        out = ctx[self]
        return gradient if out > 0 else 0


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
        return gradient, -gradient


class Mul(Binary):
    symbol = '*'

    def forward_op(self, l, r):
        return np.multiply(l, r)

    def backward(self, gradient, ctx):
        left, right = self.inputs
        return gradient * ctx[right], gradient * ctx[left]


class Div(Binary):
    symbol = '/'

    def forward_op(self, l, r):
        return np.divide(l, r)

    def backward(self, gradient, ctx):
        left, right = self.inputs
        return gradient * (1 / ctx[right]), gradient * (-ctx[left] / (ctx[right]**2))


class Pow(Binary):
    symbol = '^'

    def forward_op(self, l, r):
        return np.power(l, r)

    def backward(self, gradient, ctx):
        base_node, exponent_node = self.inputs
        base, exponent = ctx[base_node], ctx[exponent_node]
        return (gradient * exponent * np.power(base, exponent - 1),
                gradient * ctx[self] * np.log(base))


class Matmul(Binary):
    symbol = '@'

    def forward_op(self, l, r):
        return np.dot(l, r)

    def backward(self, gradient, ctx):
        left, right = self.inputs
        return np.dot(gradient, right.T), np.dot(left.T, gradient)
