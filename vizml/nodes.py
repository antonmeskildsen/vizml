import operator
import networkx as nx
import numpy as np

import graph

class Node:

    def __init__(self, value=None, derived=None):
        self.consumers = []
        self.input_values = None
        self.value = value
        self.derived = derived
    
    def __add__(self, other):
        return Add(self, other)
    
    def __sub__(self, other):
        return Sub(self, other)
    
    def __mul__(self, other):
        return Mul(self, other)
    
    def __div__(self, other):
        return Div(self, other)
    
    def __pow__(self, other):
        if isinstance(other, Node):
            return Pow(self, other)
        elif isinstance(other, int):
            return Pow(self, Constant(other))
        else:
            return NotImplemented
    
    def __call__(self, ctx={}):
        return graph.Session().run(self.graph(), ctx)
    
    def inputs(self):
        return []
    
    def compute(self, *inputs):
        pass
    
    def backward(self, a):
        pass

    def _repr_html_(self):
        return graph.of(self).draw()
    
    def graph(self):
        return graph.of(self)
    
 
class Operation(Node):

    symbol = None

    def op(self, *inputs):
        pass

    def back_op(self, val):
        pass

    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes

        for node in input_nodes:
            node.consumers.append(self)
        
        super().__init__()
    
    def inputs(self):
        return self.input_nodes

    def compute(self, *inputs):
        self.input_values = inputs
        self.value = self.op(*inputs)
        return self.value

    def backward(self, val):
        return self.back_op(val)


class Binary(Operation):

    def __init__(self, l, r):
        super().__init__([l, r])
    
    def compute(self, l_val, r_val):
        return super().compute(l_val, r_val)

    def __str__(self):
        return self.symbol


# Binary operations
class Add(Binary):

    symbol = '+'

    def op(self, l_val, r_val):
        return np.add(l_val, r_val)

    def back_op(self, val):
        return (val, val)

class Mul(Binary):

    symbol = '*'

    def op(self, l_val, r_val):
        return np.multiply(l_val, r_val)

    def back_op(self, val):
        l_val, r_val = self.input_values
        return (val*r_val, val*l_val)


#Sub = Binary.create('-', np.subtract, )
#Div = Binary.create('/', np.divide)
#Pow = Binary.create('^', np.power)
#Matmul = Binary.create('@', np.dot)




class Variable(Node):

    def __init__(self, initial_value=None, name=None):
        self.name = name
        self.input_nodes=[]

        super().__init__(initial_value)
    
    
    def compute(self):
        return self.value
    
    def backward(self, val):
        pass
    
    def __str__(self):
        if self.name is not None:
            return f'{self.name}←{self.value}'
        else:
            return str(self.value)


class Constant(Variable):

    def __init__(self, value):
        self.input_nodes=[]
        super().__init__(value)

    def __str__(self):
        return str(self.value)


class Output(Operation):

    def __init__(self, input, name=None):
        super().__init__([input])
        self.name = name
    
    def compute(self, input):
        self.value = input
        return input
    
    def backward(self, val):
        return 1
    
    def __str__(self):
        if self.name is None:
            return 'output'
        else:
            return self.name


class Input(Node):
    
    def __init__(self, name=None):
        self.name = name
        self.input_nodes=[]

        super().__init__()
    
    def compute(self, input):
        self.value = input
        return input
    
    def backward(self, val):
        return val
    
    def __str__(self):
        if self.name == None:
            return 'input'
        else:
            return self.name