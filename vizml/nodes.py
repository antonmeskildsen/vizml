import operator
import networkx as nx

class Node:

    def __init__(self):
        self.consumers = []
        self.output = None
        _default_graph.nodes.append(self)
    
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
    
    def backward(self, a):
        return [a]
    
 
class Operation(Node):

    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes

        for node in input_nodes:
            node.consumers.append(self)
        
        super().__init__()
        
    
    def compute(self, *inputs):
        pass


class Binary(Operation):

    def __init__(self, l, r, symbol, op):
        self.symbol = symbol
        self.op = op

        super().__init__([l, r])
    
    def compute(self, l_val, r_val):
        return self.op(l_val, r_val)
    
    def __str__(self):
        return self.symbol

    @staticmethod
    def create(symbol, op):
        return lambda l, r: Binary(l, r, symbol, op)


class Mul(Binary):

    def __init__(self, l, r):
        super().__init__(l, r, '*', operator.mul)
    
    def backward(self, a):
        return [Mul(self.input_nodes[1], a), Mul(self.input_nodes[0], a)]





Add = Binary.create('+', operator.add)
Sub = Binary.create('-', operator.sub)
#Mul = Binary.create('*', operator.mul)
Div = Binary.create('/', operator.truediv)
Pow = Binary.create('^', operator.pow)
Matmul = Binary.create('@', operator.matmul)




class Variable(Node):

    def __init__(self, initial_value=None, name=None):
        self.value = initial_value
        self.name = name

        super().__init__()
    
    def __str__(self):
        if self.name is not None:
            return f'{self.name}<-{self.value}'
        else:
            return str(self.value)


class Constant(Variable):

    def __init__(self, value):
        super().__init__(value)

    def __str__(self):
        return str(self.value)


class Output(Operation):

    def __init__(self, input, name=None):
        super().__init__([input])
        self.name = name
        self.value = None
    
    def compute(self, input):
        self.value = input
    
    def __str__(self):
        return self.name


class Input(Node):
    
    def __init__(self, name=None):
        self.name = name

        super().__init__()
    
    def __str__(self):
        if self.name is not None:
            return str(self.name)
        else:
            return '{}'