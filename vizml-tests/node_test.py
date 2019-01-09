import pytest
import numpy as np
from hypothesis import given
from hypothesis.strategies import tuples
from hypothesis.extra.numpy import arrays, array_shapes

from vizml import operations
from vizml import nodes


array_simple_pair = array_shapes(1, 3).flatmap(
    lambda s: tuples(arrays(np.float, s), arrays(np.float, s))
)

array_matmul_pair = array_shapes(1, 2).flatmap(
    lambda s: tuples(arrays(np.float, s), arrays(np.float, reversed(s)))
)


def generic_forward(node, func, *inputs):
    # noinspection PyCallingNonCallable
    assert node.forward_op(*inputs).all() == func(*inputs).all()


class TestUnaryOperations:

    unary_ops = [
        operations.Negative,
        operations.Log,
        operations.Sigmoid
    ]


class TestBinaryOperations:

    binary_ops = [
        operations.Add,
        operations.Sub,
        operations.Mul,
        operations.Div,
        operations.Pow,
    ]

    @pytest.mark.parametrize('BinOp', binary_ops)
    def test_binary_empty_init_raises_type_error(self, BinOp):
        with pytest.raises(TypeError, message='Expected TypeError'):
            BinOp()

    @pytest.mark.parametrize('BinOp', binary_ops)
    def test_binary_init_should_succeed(self, BinOp):
        left = nodes.Input('l')
        right = nodes.Input('r')
        BinOp(left, right)

    forward_input = [
        (operations.Add, np.add),
        (operations.Sub, np.subtract),
        (operations.Mul, np.multiply),
        (operations.Div, np.divide),
        (operations.Pow, np.power)
    ]

    # TODO: Is it wrong to test the innards like this?
    @pytest.mark.parametrize('BinOp, func', forward_input)
    @given(array_simple_pair)
    def test_binary_broadcasted_forward(self, BinOp, func, pair):
        node = BinOp(nodes.Input('l'), nodes.Input('r'))
        generic_forward(node, func, *pair)

    @given(array_matmul_pair)
    def test_matmul_forward(self, pair):
        node = operations.Matmul(nodes.Input('l'), nodes.Input('r'))
        generic_forward(node, np.dot, *pair)

    @pytest.mark.parametrize('BinOp', binary_ops)
    def test_binary_backward(self, BinOp):
        left = nodes.Input('l')
        right = nodes.Input('r')
        node = BinOp(left, right)
        ctx = {
            left: 1,
            right: 1,
            node: 1
        }
        assert len(node.backward(1, ctx)) == 2
