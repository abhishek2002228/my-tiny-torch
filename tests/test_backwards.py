"""Backward contracts for the public Tensor API."""

import numpy as np
import pytest
import weakref
import gc

from tinytorch import Tensor
from tinytorch.backends.numpy_backend import NumPyBackend


def assert_grad(tensor, expected):
    assert tensor.grad is not None
    np.testing.assert_allclose(tensor.grad.numpy(), expected, rtol=1e-5, atol=1e-6)
    assert tensor.grad.shape == tensor.shape
    assert tensor.grad.requires_grad is False


def test_shared_input_and_leaf_accumulation():
    x = Tensor([1, 2, 3], requires_grad=True)
    loss = (x * x).sum()
    loss.backward(retain_graph=True)
    assert_grad(x, [2, 4, 6])
    loss.backward()
    assert_grad(x, [4, 8, 12])
    with pytest.raises(RuntimeError, match="released"):
        loss.backward()


def test_backward_handles_deep_graph_without_recursion():
    x = Tensor(1, requires_grad=True)
    result = x
    for _ in range(1500):
        result = result + 1
    result.backward()
    assert_grad(x, np.float32(1))


def test_backward_rejects_gradient_with_wrong_input_shape():
    from tinytorch.autograd import Function

    class BadGradient(Function):
        def forward(self, array):
            return array

        def backward(self, grad_output):
            return (np.ones((2, 3), dtype=np.float32),)

    x = Tensor([[1], [2]], requires_grad=True)
    loss = BadGradient.apply(x).sum()
    with pytest.raises(ValueError, match=r"BadGradient.backward returned gradient shape \(2, 3\) for input shape \(2, 1\)"):
        loss.backward()
    assert x.grad is None


def test_graph_releases_operations_but_keeps_leaf_gradients():
    x = Tensor([1, 2], requires_grad=True)
    y = x * x
    loss = y.sum()
    operation = weakref.ref(y._ctx)
    loss.backward()
    gc.collect()
    assert operation() is None
    assert y._ctx is None
    assert loss._ctx is None
    assert_grad(x, [2, 4])


def test_retain_grad_does_not_stop_propagation():
    x = Tensor([1, 2, 3], requires_grad=True)
    z = x + 2
    z.retain_grad()
    loss = (z * z).sum()
    loss.backward(retain_graph=True)
    assert_grad(z, [6, 8, 10])
    assert_grad(x, [6, 8, 10])
    loss.backward()
    assert_grad(z, [12, 16, 20])
    assert_grad(x, [12, 16, 20])


def test_gradients_for_retained_tensors_in_branched_graph():
    x = Tensor([1.0, 2.0], requires_grad=True)
    bias = Tensor([0.5], requires_grad=True)
    scale = Tensor([2.0, 3.0], requires_grad=True)
    divisor = Tensor([4.0, 5.0], requires_grad=True)

    squared = x * x
    squared.retain_grad()
    shifted = squared + bias
    shifted.retain_grad()
    scaled = shifted * scale
    scaled.retain_grad()
    divided = scaled / divisor
    divided.retain_grad()
    output = divided + squared
    output.retain_grad()
    loss = output.sum()
    loss.retain_grad()

    loss.backward()

    assert_grad(x, [3.0, 6.4])
    assert_grad(bias, [1.1])
    assert_grad(scale, [0.375, 0.9])
    assert_grad(divisor, [-0.1875, -0.54])
    assert_grad(squared, [1.5, 1.6])
    assert_grad(shifted, [0.5, 0.6])
    assert_grad(scaled, [0.25, 0.2])
    assert_grad(divided, [1.0, 1.0])
    assert_grad(output, [1.0, 1.0])
    assert_grad(loss, np.array(1, dtype=np.float32))


@pytest.mark.parametrize("expression,expected", [
    (lambda x: x + 2, [1, 1, 1]),
    (lambda x: 2 + x, [1, 1, 1]),
    (lambda x: x - 2, [1, 1, 1]),
    (lambda x: 2 - x, [-1, -1, -1]),
    (lambda x: x * 2, [2, 2, 2]),
    (lambda x: 2 * x, [2, 2, 2]),
])
def test_scalar_arithmetic(expression, expected):
    x = Tensor([1, 2, 3], requires_grad=True)
    expression(x).sum().backward()
    assert_grad(x, expected)


@pytest.mark.parametrize("operation,left,right", [
    ("add", [1, 1, 1], [2, 2, 2]),
    ("sub", [1, 1, 1], [-2, -2, -2]),
    ("mul", [10, 20, 30], [5, 7, 9]),
])
def test_broadcast_arithmetic(operation, left, right):
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = Tensor([10, 20, 30], requires_grad=True)
    z = {"add": lambda: x + b, "sub": lambda: x - b,
         "mul": lambda: x * b, "div": lambda: x / b}[operation]()
    z.sum().backward()
    np.testing.assert_allclose(x.grad.numpy()[0], left, rtol=1e-5)
    assert_grad(b, right)


def test_scalar_tensor_and_size_one_broadcast():
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    row = Tensor([[2], [3]], requires_grad=True)
    scalar = Tensor(4, requires_grad=True)
    ((x * row) + scalar).sum().backward()
    assert_grad(row, [[6], [15]])
    assert_grad(scalar, np.array(6, dtype=np.float32))
    assert_grad(x, [[2, 2, 2], [3, 3, 3]])


@pytest.mark.parametrize("left,right", [
    ([1, 2], [3, 4]),
    ([[1, 2], [3, 4]], [5, 6]),
    ([1, 2], [[3, 4], [5, 6]]),
    ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
    ([[[1, 2], [3, 4]]], [[[5, 6], [7, 8]], [[9, 10], [11, 12]]]),
])
def test_matmul_matches_finite_difference(left, right):
    lhs = Tensor(left, requires_grad=True)
    rhs = Tensor(right, requires_grad=True)
    (lhs @ rhs).sum().backward()
    a = lhs.numpy()
    b = rhs.numpy()
    for source, actual, other, is_left in [
        (a, lhs.grad.numpy(), b, True), (b, rhs.grad.numpy(), a, False)
    ]:
        numerical = np.empty_like(source)
        for index in np.ndindex(source.shape):
            plus, minus = source.copy(), source.copy()
            plus[index] += 0.001
            minus[index] -= 0.001
            before = np.matmul(plus, other) if is_left else np.matmul(other, plus)
            after = np.matmul(minus, other) if is_left else np.matmul(other, minus)
            numerical[index] = (np.sum(before) - np.sum(after)) / 0.002
        np.testing.assert_allclose(actual, numerical, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("left_shape,right_shape", [
    ((4,), (3, 4, 2)),
    ((3, 2, 4), (4,)),
    ((1, 2, 4), (3, 4, 2)),
    ((3, 2, 4), (1, 4, 2)),
])
def test_batched_matmul_broadcast_matches_finite_difference(left_shape, right_shape):
    a = (np.arange(np.prod(left_shape), dtype=np.float32).reshape(left_shape) + 1) / 7
    b = (np.arange(np.prod(right_shape), dtype=np.float32).reshape(right_shape) + 1) / 11
    lhs = Tensor(a, requires_grad=True)
    rhs = Tensor(b, requires_grad=True)
    (lhs @ rhs).sum().backward()
    for source, actual, other, is_left in [
        (a, lhs.grad.numpy(), b, True), (b, rhs.grad.numpy(), a, False)
    ]:
        numerical = np.empty_like(source)
        for index in np.ndindex(source.shape):
            plus, minus = source.copy(), source.copy()
            plus[index] += 0.001
            minus[index] -= 0.001
            positive = np.matmul(plus, other) if is_left else np.matmul(other, plus)
            negative = np.matmul(minus, other) if is_left else np.matmul(other, minus)
            numerical[index] = (np.sum(positive) - np.sum(negative)) / 0.002
        np.testing.assert_allclose(actual, numerical, rtol=0.02, atol=0.02)


def test_stack_axis_and_repeated_input():
    x = Tensor([1, 2], requires_grad=True)
    y = Tensor([3, 4], requires_grad=True)
    Tensor.stack([x, y, x], axis=-1).sum().backward()
    assert_grad(x, [2, 2])
    assert_grad(y, [1, 1])

    scalar = Tensor(2, requires_grad=True)
    Tensor.stack([scalar, scalar]).sum().backward()
    assert_grad(scalar, np.float32(2))


@pytest.mark.parametrize("key,expected", [
    (slice(1, None), [0, 1, 1, 1]),
    ([0, 0, 2], [2, 0, 1, 0]),
    (np.array([True, False, True, False]), [1, 0, 1, 0]),
    ((None, slice(1, 3)), [0, 1, 1, 0]),
    (2, [0, 0, 1, 0]),
])
def test_indexing_scatter(key, expected):
    x = Tensor([1, 2, 3, 4], requires_grad=True)
    x[key].sum().backward()
    assert_grad(x, expected)


def test_transpose_permute_and_shape_changes():
    x = Tensor(np.arange(24).reshape(2, 3, 4), requires_grad=True)
    weights = Tensor(np.arange(24).reshape(2, 3, 4) + 1)
    ((x.transpose(0, 2).transpose(0, 2) * weights).sum()).backward()
    assert_grad(x, weights.numpy())

    x = Tensor(np.arange(24).reshape(2, 3, 4), requires_grad=True)
    weights = Tensor(np.arange(24).reshape(2, 3, 4) + 1)
    ((x.permute(2, 0, 1).permute(1, 2, 0) * weights).sum()).backward()
    assert_grad(x, weights.numpy())

    for transform in (lambda t: t.reshape(6), lambda t: t.view(6)):
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        (transform(x) * Tensor([1, 2, 3, 4, 5, 6])).sum().backward()
        assert_grad(x, [[1, 2, 3], [4, 5, 6]])


def test_expand_clone_contiguous():
    x = Tensor([1, 2, 3], requires_grad=True)
    x.expand(4, 3).clone().sum().backward()
    assert_grad(x, [4, 4, 4])

    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    x.transpose(0, 1).contiguous().sum().backward()
    assert_grad(x, [[1, 1], [1, 1]])


@pytest.mark.parametrize("axis,keepdims,expected", [
    (None, False, [[1, 1, 1], [1, 1, 1]]),
    (0, False, [[1, 1, 1], [1, 1, 1]]),
    (1, False, [[1, 1, 1], [1, 1, 1]]),
    ((0, 1), True, [[1, 1, 1], [1, 1, 1]]),
])
def test_sum_axes(axis, keepdims, expected):
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    x.sum(axis=axis, keepdims=keepdims).sum().backward()
    assert_grad(x, expected)


def test_weighted_sum_axis():
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    (x.sum(axis=1) * Tensor([2, 3])).sum().backward()
    assert_grad(x, [[2, 2, 2], [3, 3, 3]])

    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    (x.sum(axis=0) * Tensor([2, 3, 4])).sum().backward()
    assert_grad(x, [[2, 3, 4], [2, 3, 4]])


def test_sum_backward_broadcast_is_a_view_of_unchanged_gradient_storage():
    source = np.array([[10, 99], [20, 99]], dtype=np.float32)
    before = source.copy()
    incoming = source[:, 0]  # Noncontiguous input gradient, shape (2,).
    result = NumPyBackend().broadcast_reduced_grad(
        incoming, (2, 3), axis=1
    )

    np.testing.assert_array_equal(result, [[10, 10, 10], [20, 20, 20]])
    np.testing.assert_array_equal(source, before)
    assert result.shape == (2, 3)
    assert result.strides == (incoming.strides[0], 0)
    assert np.shares_memory(result, incoming)
    assert result.ctypes.data == incoming.ctypes.data
    assert not result.flags.owndata
    assert not result.flags.writeable


def test_mean_axes_and_scalar_sum():
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    (x.mean(axis=1) * Tensor([2, 3])).sum().backward()
    assert_grad(x, [[2 / 3] * 3, [1] * 3])
    scalar = Tensor(2, requires_grad=True)
    scalar.sum(axis=0).mean().backward()
    assert_grad(scalar, np.array(1, dtype=np.float32))

    scalar = Tensor(2, requires_grad=True)
    scalar.mean().backward()
    assert_grad(scalar, np.float32(1))

    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    x.mean(axis=(0, 1), keepdims=True).sum().backward()
    assert_grad(x, [[1 / 6] * 3, [1 / 6] * 3])


def test_max_ties_share_gradient():
    x = Tensor([[2, 2, 1], [3, 1, 3]], requires_grad=True)
    (x.max(axis=1) * Tensor([2, 3])).sum().backward()
    assert_grad(x, [[1, 1, 0], [1.5, 0, 1.5]])
    x = Tensor([2, 2, 1], requires_grad=True)
    x.max().backward()
    assert_grad(x, [0.5, 0.5, 0])

    scalar = Tensor(2, requires_grad=True)
    scalar.max(axis=0).backward()
    assert_grad(scalar, np.float32(1))

    x = Tensor([[2, 2, 1], [3, 1, 3]], requires_grad=True)
    x.max(axis=1, keepdims=True).sum().backward()
    assert_grad(x, [[0.5, 0.5, 0], [0.5, 0, 0.5]])


def test_multiple_axis_max_and_indexing():
    x = Tensor([[[1, 3], [3, 0]], [[2, 3], [1, 3]]], requires_grad=True)
    x.max(axis=(0, 2)).sum().backward()
    assert_grad(x, [[[0, 0.5], [0.5, 0]], [[0, 0.5], [0, 0.5]]])

    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    x[[0, 0], [1, 1]].sum().backward()
    assert_grad(x, [[0, 2], [0, 0]])


def test_constructed_tensor_is_fresh_leaf_but_stack_is_connected():
    x = Tensor([1, 2], requires_grad=True)
    Tensor([x, x], requires_grad=True).sum().backward()
    assert x.grad is None
    Tensor.stack([x, x]).sum().backward()
    assert_grad(x, [2, 2])


def test_division_of_two_tensors():
    x = Tensor([2, 6], requires_grad=True)
    y = Tensor([2, 3], requires_grad=True)
    (x / y).sum().backward()
    assert_grad(x, [0.5, 1 / 3])
    assert_grad(y, [-0.5, -2 / 3])


@pytest.mark.parametrize("expression,expected", [
    (lambda x: x / 2, [0.5, 0.5, 0.5]),
    (lambda x: 2 / x, [-2, -0.5, -2 / 9]),
])
def test_division_with_python_scalar_in_either_position(expression, expected):
    x = Tensor([1, 2, 3], requires_grad=True)
    expression(x).sum().backward()
    assert_grad(x, expected)


def test_division_with_scalar_tensor():
    x = Tensor([2, 4, 6], requires_grad=True)
    scale = Tensor(2, requires_grad=True)
    (x / scale).sum().backward()
    assert_grad(x, [0.5, 0.5, 0.5])
    assert_grad(scale, np.float32(-3))


def test_division_broadcasts_and_reduces_both_gradients():
    x = Tensor([[2, 4, 6], [8, 10, 12]], requires_grad=True)
    y = Tensor([2, 4, 6], requires_grad=True)
    (x / y).sum().backward()
    assert_grad(x, [[0.5, 0.25, 1 / 6], [0.5, 0.25, 1 / 6]])
    assert_grad(y, [-2.5, -0.875, -0.5])


def test_untracked_denominator_does_not_receive_gradient():
    x = Tensor([2, 4], requires_grad=True)
    y = Tensor([2, 2])
    (x / y).sum().backward()
    assert_grad(x, [0.5, 0.5])
    assert y.grad is None


def test_division_agrees_with_finite_differences():
    x = Tensor([2, 3], requires_grad=True)
    y = Tensor([4, 5], requires_grad=True)
    (x / y).sum().backward()
    for tensor, other, is_left in [(x, y, True), (y, x, False)]:
        values = tensor.numpy()
        numeric = np.empty_like(values)
        for index in np.ndindex(values.shape):
            plus = values.copy()
            minus = values.copy()
            plus[index] += 0.001
            minus[index] -= 0.001
            positive = np.sum(plus / other.numpy() if is_left else other.numpy() / plus)
            negative = np.sum(minus / other.numpy() if is_left else other.numpy() / minus)
            numeric[index] = (positive - negative) / 0.002
        np.testing.assert_allclose(tensor.grad.numpy(), numeric, rtol=0.002, atol=0.002)
