from typing import List, Optional
from abc import ABC, abstractmethod


class Node(ABC):

    def __init__(self, input_nodes=None):
        if input_nodes is None:
            self.inputs = []
        else:
            self.inputs = input_nodes

        self.consumers = []

    @abstractmethod
    def forward(self, *inputs, **placeholders):
        ...

    @abstractmethod
    def backward(self, gradient, ctx):
        ...


class Variable(Node):

    def __init__(self, value, name):
        super().__init__()
        self.name = name
        self.value = value

    def forward(self):
        return self.value

    def backward(self, gradient, ctx):
        return gradient  # TODO: Should you even call backward on variable?

    def __str__(self):
        return str(self.name)


class Constant(Variable):

    def __init__(self, value):
        super().__init__(value, None)

    def __str__(self):
        return str(self.value)


class Output(Node):

    def __init__(self, input, name='output'):
        super().__init__([input])
        self.name = name

    def forward(self, input):
        return input

    def backward(self, gradient, ctx):
        return 1

    def __str__(self):
        return self.name


class Input(Node):

    def __init__(self, name):
        self.name = name
        super().__init__()

    def forward(self, **placeholders):
        return placeholders[self.name]

    def backward(self, gradient, ctx):
        return gradient  # TODO: Necessary?
