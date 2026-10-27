"""Check operation results and storage behavior independently of Tensor wiring."""

import numpy as np
import pytest

from tinytorch import Tensor
from tinytorch.ops import (
    Add, Sub, Mul, Div, MatMul, Stack, GetItem, Transpose, Permute,
    Reshape, View, Expand, Clone, Contiguous, Sum, Mean, Max,
)


@pytest.mark.parametrize("operation, expected", [
    (Add, [3, 6]), (Sub, [1, 2]), (Mul, [2, 8]), (Div, [2, 2]),
])
def test_arithmetic(operation, expected):
    output = operation.apply(Tensor([2, 4]), Tensor([1, 2]))
    np.testing.assert_allclose(output.numpy(), expected)


def test_scalar_results_and_operand_order():
    # Typed scalars preserve float32 even under NumPy 1.x scalar promotion.
    assert Sub.apply(np.float32(10), Tensor(3)).numpy() == 7
    assert Div.apply(np.float32(12), Tensor(3)).numpy() == 4
    for operation in (Add, Sub, Mul, Div):
        assert operation.apply(Tensor(2), np.float32(2)).shape == ()
    output = MatMul.apply(Tensor([1, 2]), Tensor([3, 4]))
    assert output.shape == ()
    assert output.numpy() == 11


@pytest.mark.parametrize("operation, numpy_op", [
    (Add, np.add), (Sub, np.subtract), (Mul, np.multiply), (Div, np.divide),
])
def test_python_scalar_promotion_matches_installed_numpy(operation, numpy_op):
    x = Tensor(2)
    expected = np.asarray(numpy_op(x._storage, 2))
    if expected.dtype == np.dtype(np.float32):
        result = operation.apply(x, 2)
        np.testing.assert_array_equal(result.numpy(), expected)
        assert result.shape == ()
    else:
        with pytest.raises(NotImplementedError, match="Unsupported storage dtype"):
            operation.apply(x, 2)


def test_views_and_copies():
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    transposed = Transpose.apply(x, dim0=-1, dim1=-2)
    permuted = Permute.apply(x, axes=(1, 0))
    sliced = GetItem.apply(x, key=(slice(1, None), slice(None, 2)))
    reshaped = Reshape.apply(x, shape=(3, 2))
    for output in (transposed, permuted, sliced, reshaped):
        assert np.shares_memory(output._storage, x._storage)
        assert output._ctx.inputs == (x,)
    for output in (
        Clone.apply(x), Contiguous.apply(transposed),
        Reshape.apply(transposed, shape=(6,)),
        GetItem.apply(x, key=[0, 1]), Stack.apply(x, x, axis=0),
    ):
        assert not np.shares_memory(output._storage, x._storage)
    assert GetItem.apply(x, key=(0, 0)).shape == ()
    assert Contiguous.apply(Tensor(1)).shape == ()


def test_strict_view():
    array = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
    x = Tensor._from_storage(array)
    flat = View.apply(x, shape=(-1,))
    assert np.shares_memory(flat._storage, array)
    assert flat.strides == (8,)
    transposed = Transpose.apply(Tensor(array), dim0=0, dim1=1)
    with pytest.raises(ValueError):
        View.apply(transposed, shape=(-1,))
    assert View.apply(Tensor([]), shape=(0, 2)).shape == (0, 2)


def test_broadcast_view_and_arithmetic():
    x = Tensor([1, 2, 3])
    expanded = Expand.apply(x, shape=(2, 3))
    assert expanded.strides == (0, 4)
    assert np.shares_memory(expanded._storage, x._storage)
    assert not expanded._storage.flags.writeable
    np.testing.assert_array_equal(
        Add.apply(Tensor([[10], [20]]), x).numpy(),
        [[11, 12, 13], [21, 22, 23]],
    )
    with pytest.raises(ValueError):
        Expand.apply(x, shape=(-1, 3))


@pytest.mark.parametrize("operation, numpy_operation", [
    (Sum, np.sum), (Mean, np.mean), (Max, np.max),
])
def test_reductions(operation, numpy_operation):
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    scalar = operation.apply(x)
    assert scalar.shape == ()
    assert scalar.numpy() == numpy_operation(x._storage)
    output = operation.apply(x, axis=1, keepdims=True)
    np.testing.assert_array_equal(
        output.numpy(), numpy_operation(x._storage, axis=1, keepdims=True),
    )
