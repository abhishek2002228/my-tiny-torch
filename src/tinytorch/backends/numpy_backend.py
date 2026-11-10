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

    def ones_like(self, array):
        return np.ones_like(array)

    def zeros_like(self, array):
        return np.zeros_like(array)

    def scalar_like(self, value, array):
        """Represent a Python scalar in a tensor operand's storage dtype."""
        return np.asarray(value, dtype=array.dtype)

    def broadcast_reduced_grad(self, grad, shape, *, axis=None, keepdims=False):
        """Undo a forward pass reduction by inserting removed axes and broadcasting."""
        if axis is not None and not keepdims and len(shape):
            axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
            axes = tuple(item % len(shape) for item in axes)
            grad = np.expand_dims(grad, axis=axes)
        return self.expand(grad, shape=shape)

    def reduction_count(self, shape, *, axis=None):
        """Counts - How many values did the forward pass reduce ?"""
        if not shape:
            return 1
        if axis is None:
            return int(np.prod(shape))
        axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
        count = 1
        for item in axes:
            count *= shape[item]
        return count

    def max_grad(self, array, maxima, grad, *, axis=None, keepdims=False):
        """Split a maximum's gradient equally among tied maxima."""
        maxima = self.broadcast_reduced_grad(
            maxima, array.shape, axis=axis, keepdims=keepdims
        )
        mask = array == maxima
        counts = self.sum(mask, axis=axis, keepdims=True).astype(array.dtype, copy=False)
        expanded = self.broadcast_reduced_grad(
            grad, array.shape, axis=axis, keepdims=keepdims
        )
        return self.div(self.mul(expanded, mask), counts)

    def scatter_add(self, shape, key, grad, *, dtype):
        """Reverse indexing, adding repeated advanced-index contributions."""
        result = np.zeros(shape, dtype=_NUMPY_DTYPES[dtype])
        np.add.at(result, key, grad)
        return result

    def unstack(self, grad, *, axis):
        moved = np.moveaxis(grad, axis, 0)
        return tuple(
            moved[index:index + 1].reshape(moved.shape[1:])
            for index in range(moved.shape[0])
        )

    def matmul_grad(self, lhs, rhs, grad, *, need_lhs=True, need_rhs=True):
        """Reverse NumPy matmul's vector promotion and batch broadcasting."""
        left_vector, right_vector = lhs.ndim == 1, rhs.ndim == 1
        left = lhs[np.newaxis, :] if left_vector else lhs
        right = rhs[:, np.newaxis] if right_vector else rhs
        if left_vector and right_vector:
            grad = self.reshape(grad, shape=(1, 1))
        elif left_vector:
            grad = self.reshape(grad, shape=grad.shape[:-1] + (1, grad.shape[-1]))
        elif right_vector:
            grad = self.reshape(grad, shape=grad.shape + (1,))

        left_grad = right_grad = None
        if need_lhs:
            left_grad = self.matmul(grad, np.swapaxes(right, -1, -2))
            if left_vector:
                left_grad = np.squeeze(left_grad, axis=-2)
            left_grad = self.reduce_broadcasted_grad(left_grad, lhs.shape)
        if need_rhs:
            right_grad = self.matmul(np.swapaxes(left, -1, -2), grad)
            if right_vector:
                right_grad = np.squeeze(right_grad, axis=-1)
            right_grad = self.reduce_broadcasted_grad(right_grad, rhs.shape)
        return left_grad, right_grad


    def reduce_broadcasted_grad(self, grad, shape):
        """Reduce gradients for tensors which were broadcasted in the forward pass"""
        # Sum over leading dimensions added by broadcasting.
        while grad.ndim > len(shape):
            grad = self.sum(grad, axis=0)

        # Sum over dimensions expanded from size 1.
        for axis, size in enumerate(shape):
            if size == 1 and grad.shape[axis] != 1:
                grad = self.sum(grad, axis=axis, keepdims=True)

        if grad.shape != shape:
            raise ValueError(f"Cannot reduce gradient of shape {grad.shape} to {shape}")
        
        return grad
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
