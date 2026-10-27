"""CPU numerical operations implemented with NumPy.

Methods return ndarrays, retain NumPy dtype behavior, and preserve view storage
where the operation permits it. No device transfers happen here.

 The NumPy backend converts directly to the requested dtype without preliminary value validation.
 Numeric strings and convertible object arrays are accepted;
 booleans become 1.0 or 0.0 in float32 storage. NumPy conversion errors and warnings propagate unchanged.
"""

import numpy as np

_NUMPY_DTYPES = {
    "float32": np.dtype(np.float32),
}

class NumPyBackend:
    device = "cpu"
    storage_type = np.ndarray

    @property
    def supported_dtypes(self):
        return frozenset(_NUMPY_DTYPES)

    def create(self, data, *, dtype="float32"):
        if dtype not in _NUMPY_DTYPES:
            raise NotImplementedError(f"Unsupported dtype: {dtype}")

        numpy_dtype = _NUMPY_DTYPES[dtype]

        if (
            not isinstance(data, (list, tuple, np.ndarray))
            and not np.isscalar(data)
        ):
            raise TypeError("Unsupported input type for Tensor construction.")

        return np.array(data, dtype=numpy_dtype, copy=True, order="C")

    def validate_storage(self, storage):
        if not isinstance(storage, np.ndarray):
            raise TypeError("Numpy backend storage must be an ndarray.")

        if storage.dtype not in _NUMPY_DTYPES.values():
            raise NotImplementedError(f"Unsupported storage dtype: {storage.dtype}")

    def shape(self, storage):
        """Return the logical dimension sizes without copying storage."""
        return storage.shape

    def dtype(self, storage):
        """Return the dtype name used by the framework, such as 'float32'."""
        return storage.dtype.name

    def strides(self, storage):
        """Return strides in bytes, including zero and negative strides."""
        return storage.strides

    def numel(self, storage):
        """Count logical elements: one for a scalar, zero for empty storage."""
        return storage.size

    def is_contiguous(self, storage):
        """Use NumPy's C-contiguity check, including singleton and empty axes."""
        return storage.flags.c_contiguous

    def to_numpy(self, storage, *, copy=True):
        """Export an independent C-order copy by default.

        copy=False is an internal option that returns the original storage.
        Tensor.numpy always requests a copy.
        """
        return storage.copy(order="C") if copy else storage

    def add(self, lhs, rhs):
        return np.asarray(lhs + rhs)

    def sub(self, lhs, rhs):
        return np.asarray(lhs - rhs)

    def mul(self, lhs, rhs):
        return np.asarray(lhs * rhs)

    def div(self, lhs, rhs):
        return np.asarray(lhs / rhs)

    def matmul(self, lhs, rhs):
        return np.asarray(np.matmul(lhs, rhs))

    def stack(self, *arrays, axis=0):
        return np.stack(arrays, axis=axis)

    def getitem(self, array, *, key):
        return np.asarray(array[key])

    def transpose(self, array, *, dim0, dim1):
        return np.swapaxes(array, dim0, dim1)

    def permute(self, array, *, axes):
        return np.transpose(array, axes=axes)

    def reshape(self, array, *, shape):
        return np.reshape(array, shape)

    def view(self, array, *, shape):
        if np.lib.NumpyVersion(np.__version__) >= "2.1.0":
            return np.reshape(array, shape, copy=False)
        # Compatibility with NumPy 1.24–2.0, which lack reshape(copy=False).
        # Changing shape on a separate ndarray view cannot copy element storage.
        # Unlike shares_memory checks, this also handles empty arrays correctly.
        result = array.view()
        try:
            result.shape = shape
        except AttributeError as error:
            raise ValueError(
                "Requested view requires copying; use reshape instead."
            ) from error
        return result

    def expand(self, array, *, shape):
        # NumPy rejects negative dimensions and incompatible broadcast shapes.
        return np.broadcast_to(array, shape)

    def clone(self, array):
        return array.copy(order="K")

    def contiguous(self, array):
        # ascontiguousarray would turn a 0D array into a 1D array.
        return array if array.flags.c_contiguous else array.copy(order="C")

    def sum(self, array, *, axis=None, keepdims=False):
        return np.asarray(np.sum(
            array, axis=axis,
            keepdims=keepdims,
        ))

    def mean(self, array, *, axis=None, keepdims=False):
        return np.asarray(np.mean(
            array, axis=axis,
            keepdims=keepdims,
        ))

    def max(self, array, *, axis=None, keepdims=False):
        return np.asarray(np.max(
            array, axis=axis,
            keepdims=keepdims,
        ))

    # Memory Profiling

    def nbytes(self, storage):
        return storage.nbytes

    def memory_info(self, storage):
        owner = storage

        while isinstance(owner.base, np.ndarray):
            owner = owner.base

        return {
            "logical_bytes": storage.nbytes,
            "backing_bytes": (
                owner.nbytes if owner.flags.owndata else None
            ),
            "is_view": storage.base is not None,
        }