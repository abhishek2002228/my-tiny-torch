"""Device selection and backend execution through the public Tensor API."""

import numpy as np
import pytest

from tinytorch import Tensor
from tinytorch.backends import get_backend
from tinytorch.backends.numpy_backend import NumPyBackend
from tinytorch import autograd


def test_cpu_operation_records_device_and_backend():
    x = Tensor([1, 2], requires_grad=True)
    result = x + 2
    assert result._ctx.device == "cpu"
    assert result._ctx.backend is get_backend("cpu")
    assert isinstance(result._ctx.backend, NumPyBackend)
    assert result._ctx.inputs == (x, 2)


def test_apply_uses_selected_backend_without_copying_its_result(monkeypatch):
    selected_devices = []
    calls = []
    result_array = np.array([9, 10], dtype=np.float32)

    class RecordingBackend:
        storage_type = np.ndarray

        def add(self, lhs, rhs):
            calls.append((lhs, rhs))
            return result_array

    backend = RecordingBackend()

    def select(device):
        selected_devices.append(device)
        return backend

    monkeypatch.setattr(autograd, "get_backend", select)
    x = Tensor([1, 2])
    output = 2 + x
    assert selected_devices == ["cpu"]
    assert calls[0][0] == 2
    assert calls[0][1] is x._storage
    assert output._storage is result_array
    assert output._ctx is None


def test_unknown_device_has_no_cpu_fallback():
    with pytest.raises(NotImplementedError, match="No backend"):
        get_backend("cuda")


def test_mixed_devices_rejected_before_backend_selection(monkeypatch):
    x, y = Tensor([1]), Tensor([2])
    # Simulate inconsistent inputs without pretending to implement GPU storage.
    y._device = "cuda"

    def unexpected_selection(device):
        pytest.fail("Mixed devices must be rejected before backend selection.")

    monkeypatch.setattr(autograd, "get_backend", unexpected_selection)
    with pytest.raises(ValueError, match="same device"):
        x + y


def test_backend_dispatch_preserves_transpose_views():
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    output = x.transpose(0, 1)
    assert output._ctx.backend is get_backend("cpu")
    assert output.strides == (4, 12)
    assert np.shares_memory(output._storage, x._storage)
