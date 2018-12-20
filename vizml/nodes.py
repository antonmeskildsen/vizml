import operator
import networkx as nx
import numpy as np

import graph

class Node:

    def __init__(self, value=None):
        self.consumers = []
        #self.input_values = None
        self.value = value
        self.gradient_values=None
    
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
        return self.forward(ctx)
    
    def inputs(self):
        return []

    @property
    def input_values(self):
        return [n.value for n in self.input_nodes]
    
    @property
    def consumer_gradients(self):
        if len(self.consumers) == 0:
            output_grad = 1
        else:
            output_grad = 0
            for c in self.consumers:
                grad = c.gradient_values

                if len(c.input_nodes) == 1:
                    output_grad += c.gradient_values
                else:
                    idx = c.input_nodes.index(self)
                    output_grad += c.gradient_values[idx]

        return output_grad
    
    def compute(self, ctx):
        pass
    
    def compute_backward(self, ctx):
        pass

    def reset_all(self):
        for node in self.graph.topological():
            node.reset_value()
    
    def reset_value(self):
        self.value = None

    def forward(self, ctx={}):
        for node in self.graph.topological():
            node.compute(ctx)
        
        return self.value

    def backward(self, ctx={}):
        for node in self.graph.topological_reverse():
            node.compute_backward(ctx)

    def _repr_html_(self):
        return graph.of(self).draw()
    
    def show(self, values=False, gradients=False):
        g = graph.of(self)
        g.show_values = values
        g.show_gradients = gradients
        return g
    
    @property
    def graph(self):
        return graph.of(self)
    
 
class Operation(Node):

    symbol = ''

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

    def compute(self, ctx):
        self.value = self.op(*self.input_values)
        return self.value
    
    def compute_backward(self, ctx):
        self.gradient_values = self.back_op(self.consumer_gradients)
        return self.gradient_values


class Binary(Operation):

    def __init__(self, l, r):
        super().__init__([l, r])

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

class Pow(Binary):

    symbol = '^'

    def op(self, l_val, r_val):
        return np.power(l_val, r_val)
    
    def back_op(self, val):
        base, exponent = self.input_values
        return (val*exponent*np.power(base, exponent-1), val*self.value*np.log(base))


#Sub = Binary.create('-', np.subtract, )
#Div = Binary.create('/', np.divide)
#Pow = Binary.create('^', np.power)
#Matmul = Binary.create('@', np.dot)




class Variable(Node):

    def __init__(self, initial_value=None, name=None):
        self.name = name
        self.input_nodes=[]

        super().__init__(initial_value)
    
    
    def compute(self, ctx):
        return self.value
    
    def compute_backward(self, ctx):
        self.gradient_values = self.consumer_gradients

    def reset_value(self):
        pass
    
    def __str__(self):
        if self.name is not None:
            return str(self.name)
        else:
            return 'var'


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
    
    def compute(self, ctx):
        self.value = self.input_nodes[0].value
        return self.value
    
    def compute_backward(self, ctx):
        self.gradient_values = 1
        return self.gradient_values
    
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
    
    def compute(self, ctx):
        self.value = ctx[self.name]
        return self.value
    
    def compute_backward(self, ctx):
        self.gradient_values = self.consumer_gradients
        return self.gradient_values
    
    def __str__(self):
        if self.name == None:
            return 'input'
        else:
            return self.name