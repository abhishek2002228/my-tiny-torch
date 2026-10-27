"""Operation setup and history recording, not gradient calculation."""

import numpy as np
import pytest

from tinytorch import Tensor
from tinytorch.autograd import Function
from tinytorch.ops import Add


@pytest.mark.parametrize("call, operation", [
    (lambda x: x + 2, "Add"),
    (lambda x: 2 + x, "Add"),
    (lambda x: x - 2, "Sub"),
    (lambda x: 2 - x, "Sub"),
    (lambda x: x * 2, "Mul"),
    (lambda x: 2 * x, "Mul"),
    (lambda x: x / 2, "Div"),
    (lambda x: 2 / x, "Div"),
    (lambda x: x @ x, "MatMul"),
    (lambda x: x[:, :1], "GetItem"),
    (lambda x: x.transpose(0, 1), "Transpose"),
    (lambda x: x.permute((1, 0)), "Permute"),
    (lambda x: x.reshape(4), "Reshape"),
    (lambda x: x.view(4), "View"),
    (lambda x: x.expand(2, 2), "Expand"),
    (lambda x: x.clone(), "Clone"),
    (lambda x: x.transpose(0, 1).contiguous(), "Contiguous"),
    (lambda x: x.sum(), "Sum"),
    (lambda x: x.mean(), "Mean"),
    (lambda x: x.max(), "Max"),
    (lambda x: Tensor.stack([x, x]), "Stack"),
])
@pytest.mark.parametrize("track", [False, True])
def test_public_operations_route_through_apply(call, operation, track):
    x = Tensor([[1, 2], [3, 4]], requires_grad=track)
    output = call(x)
    assert output.requires_grad is track
    assert output.device == x.device
    if track:
        assert type(output._ctx).__name__ == operation
    else:
        assert output._ctx is None


@pytest.mark.parametrize("left, right", [(False, False), (True, False), (False, True), (True, True)])
def test_any_tensor_input_can_request_history(left, right):
    a, b = Tensor([1], requires_grad=left), Tensor([2], requires_grad=right)
    output = a + b
    assert output.requires_grad is (left or right)
    if output.requires_grad:
        assert output._ctx.inputs[0] is a
        assert output._ctx.inputs[1] is b
    else:
        assert output._ctx is None


def test_context_is_per_call_and_parameters_belong_to_that_call():
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    rows, columns = x.sum(axis=1), x.sum(axis=0, keepdims=True)
    assert rows._ctx is not columns._ctx
    assert rows._ctx.axis == 1
    assert columns._ctx.axis == 0
    assert columns._ctx.keepdims is True
    assert rows._ctx.inputs[0] is columns._ctx.inputs[0] is x


def test_copying_construction_breaks_history_but_clone_and_stack_retain_it():
    source = Tensor([1, 2], requires_grad=True) * 2
    for leaf in (Tensor(source, requires_grad=True), Tensor([source, source], requires_grad=True)):
        assert leaf.requires_grad
        assert leaf._ctx is None
    clone = source.clone()
    stack = Tensor.stack([source, source])
    assert clone._ctx.inputs[0] is source
    assert stack._ctx.inputs[0] is stack._ctx.inputs[1] is source
    assert not np.shares_memory(clone._storage, source._storage)
    assert not np.shares_memory(stack._storage, source._storage)


def test_graph_connections_are_retained_through_a_chain():
    x = Tensor([1, 2], requires_grad=True)
    scaled = 2 * x
    shifted = scaled + 1
    loss = shifted.sum()
    assert loss._ctx.inputs[0] is shifted
    assert shifted._ctx.inputs[0] is scaled
    assert scaled._ctx.inputs[0] == 2
    assert scaled._ctx.inputs[1] is x


def test_forward_is_required_but_backward_is_optional():
    with pytest.raises(TypeError):
        Function()

    class Identity(Function):
        def forward(self, storage):
            return storage

    x = Tensor([1], requires_grad=True)
    output = Identity.apply(x)
    assert output._storage is x._storage
    with pytest.raises(NotImplementedError, match="backward"):
        output._ctx.backward(np.ones(1, dtype=np.float32))


def test_forward_must_return_backend_storage():
    class ScalarResult(Function):
        def forward(self, storage):
            return np.float32(1)

    with pytest.raises(TypeError, match="Forward must return"):
        ScalarResult.apply(Tensor(1))


@pytest.mark.parametrize("operands", [(), (1, 2)])
def test_apply_requires_a_tensor_input(operands):
    with pytest.raises(ValueError, match="At least one"):
        Add.apply(*operands)


@pytest.mark.parametrize("value", ["2", b"2", 2j, [2], np.array([2], dtype=np.float32), None])
def test_arithmetic_does_not_accept_construction_inputs(value):
    with pytest.raises(TypeError, match="Unsupported input type"):
        Tensor([1, 2]) + value


def test_numpy_promotion_is_not_silently_cast_back():
    x = Tensor([1, 2])
    with pytest.raises(NotImplementedError, match="Unsupported storage dtype"):
        x + np.float64(1e100)
