"""CPU storage allocation, validation, metadata, and export contracts."""

import numpy as np
import pytest

from tinytorch.backends.numpy_backend import NumPyBackend


@pytest.fixture
def backend():
    return NumPyBackend()


@pytest.mark.parametrize("data", [
    2, "3.5", [1, "2"], [True, False], [], [[], []],
    np.array(["1", "2"], dtype=object),
])
def test_create_converts_values_and_allocates_cpu_float32(backend, data):
    storage = backend.create(data, dtype="float32")
    np.testing.assert_array_equal(storage, np.array(data, dtype=np.float32))
    assert storage.dtype == np.dtype(np.float32)
    assert storage.flags.c_contiguous


def test_create_from_noncontiguous_storage_copies_and_packs(backend):
    source = np.arange(12, dtype=np.float32).reshape(3, 4).T
    storage = backend.create(source)
    assert storage is not source
    assert not np.shares_memory(storage, source)
    assert storage.flags.c_contiguous
    np.testing.assert_array_equal(storage, source)


def test_creation_and_storage_validation_use_backend_supported_dtypes(backend):
    assert backend.supported_dtypes == frozenset({"float32"})
    with pytest.raises(NotImplementedError):
        backend.create([1], dtype="float64")
    with pytest.raises(NotImplementedError):
        backend.validate_storage(np.array([1], dtype=np.float64))
    with pytest.raises(TypeError):
        backend.validate_storage([1])


def test_dtype_support_is_derived_from_backend_mapping(monkeypatch, backend):
    from tinytorch.backends import numpy_backend
    from tinytorch import Tensor

    monkeypatch.setitem(numpy_backend._NUMPY_DTYPES, "float64", np.dtype(np.float64))
    assert "float64" in backend.supported_dtypes
    tensor = Tensor([1, 2], dtype="float64")
    assert tensor.dtype == "float64"
    result = tensor + tensor
    assert result.dtype == "float64"
    np.testing.assert_array_equal(result.numpy(), [2, 4])


@pytest.mark.parametrize("storage", [
    np.array(3, dtype=np.float32),
    np.empty((0, 2), dtype=np.float32),
    np.arange(6, dtype=np.float32).reshape(2, 3).T,
    np.arange(6, dtype=np.float32)[::-1],
    np.broadcast_to(np.array([1, 2], dtype=np.float32), (3, 2)),
])
def test_metadata_and_export_preserve_actual_layout(backend, storage):
    before = (storage.shape, storage.strides, storage.flags.writeable)
    backend.validate_storage(storage)
    assert backend.shape(storage) == storage.shape
    assert backend.dtype(storage) == "float32"
    assert backend.strides(storage) == storage.strides
    assert backend.numel(storage) == storage.size
    assert backend.is_contiguous(storage) == storage.flags.c_contiguous
    exported = backend.to_numpy(storage, copy=True)
    np.testing.assert_array_equal(exported, storage)
    assert exported is not storage
    assert not np.shares_memory(exported, storage)
    assert exported.flags.c_contiguous
    assert exported.flags.writeable
    assert backend.to_numpy(storage, copy=False) is storage
    assert (storage.shape, storage.strides, storage.flags.writeable) == before


@pytest.mark.parametrize("data", [["hello"], [[1], [2, 3]]])
def test_numpy_conversion_errors_propagate(backend, data):
    with pytest.raises(ValueError):
        backend.create(data)
