import pytest
import numpy as np
from hypothesis import given
from hypothesis.strategies import tuples
from hypothesis.extra.numpy import arrays, array_shapes

from vizml import operations
from vizml import nodes


def compatible_shapes(left, right):
    l_shape = left.shape
    r_shape = right.shape

    l_idx = 0
    r_idx = 0
    while len(l_shape) > l_idx and len(r_shape) > r_idx:
        if l_shape[l_idx] == r_shape[r_idx]:
            l_idx += 1
            r_idx += 1
        elif l_shape[l_idx] == 1:
            l_idx += 1
        elif r_shape[r_idx] == 1:
            r_idx += 1

    return len(l_shape) == l_idx and len(r_shape) == r_idx


array_single = array_shapes(1, 3)
array_simple_pair = array_shapes(1, 3).flatmap(
    lambda s: tuples(arrays(np.float, s), arrays(np.float, s))
)

array_matmul_pair = array_shapes(2, 2).flatmap(
    lambda s: tuples(arrays(np.float, s), arrays(np.float, reversed(s)))
)


class TestOperations:

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

    @pytest.mark.parametrize('BinOp, func', forward_input)
    @given(array_simple_pair)
    def test_binary_broadcasted_forward(self, BinOp, func, pair):
        print(pair)
        l_val, r_val = pair
        node = BinOp(nodes.Input('l'), nodes.Input('r'))

        # noinspection PyCallingNonCallable
        assert node.forward_op(l_val, l_val).all() == func(l_val, l_val).all()
