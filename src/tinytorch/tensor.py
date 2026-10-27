"""
Tensor implementation for my-tiny-torch. Based on the device, we can choose different backends
"""

from __future__ import annotations
import numpy as np
from .ops import (
    Add, Sub, Mul, Div, MatMul, Stack, GetItem, Transpose, Permute,
    Reshape, View, Expand, Clone, Contiguous, Sum, Mean, Max,
)

from .backends import get_backend
from .backends import _BACKENDS


class Tensor:
    """
    Numerical values and the metadata describing their storage.
    """

    def __init__(self, data, requires_grad=False, dtype="float32", device="cpu"):
        """Copy input into independent, contiguous storage via the backend.

        Accept another Tensor, a nonempty list/tuple of equally shaped Tensors,
        or ordinary input convertible by the selected backend.

        Construction copies values and starts fresh gradient history, including
        when stacking input Tensors. requires_grad belongs to this new leaf;
        it does not connect the leaf to the inputs.
        """
        if device not in _BACKENDS:
            raise NotImplementedError(f"No backend is available for device {device!r}.")

        self._backend = get_backend(device)

        if dtype not in self._backend.supported_dtypes:
            raise NotImplementedError(
                f"Backend for {device!r} does not support dtype {dtype!r}."
            )

        self._device = device
        self._requires_grad = requires_grad
        self._ctx = None  # Context for autograd; set when this tensor is the output of an operation.

        # create fresh storage using backend
        if isinstance(data, Tensor):
            if data.device != device:
                raise NotImplementedError("Cross-device copying is not implemented yet.")
            data = data._storage
        elif (
            isinstance(data, (list, tuple))
            and len(data) > 0
            and all(isinstance(t, Tensor) for t in data)
        ):
            if any (t.device != device for t in data):
                raise NotImplementedError("Cross-device construction is not implemented yet.")
            data = [t._storage for t in data]

        self._storage = self._backend.create(data, dtype=dtype)


    @classmethod
    def _from_storage(cls, storage, *, device = "cpu", requires_grad = False) -> Tensor:
        """Wrap an existing storage without invoking copying construction.

        Retain the exact storage, including noncontiguous/read-only views.
        Reject incompatible input instead of silently casting or copying it.
        """

        if device not in _BACKENDS:
            raise NotImplementedError(f"No backend is available for device {device!r}.")

        #check if passed storage matches the backend
        backend = get_backend(device)
        backend.validate_storage(storage)

        # Create the Python object without running the copying __init__.
        tensor = cls.__new__(cls)
        tensor._storage = storage
        tensor._backend = backend
        tensor._device = device
        tensor._requires_grad = requires_grad
        tensor._ctx = None  # Context for autograd; set when this tensor is the output of an operation.
        return tensor

    @classmethod
    def stack(cls, tensors: list[Tensor], axis=0) -> Tensor:
        """Stack equally shaped tensors along a new axis into independent storage.

        Unlike constructor stacking, this operation will retain input gradient
        connections when tracking is enabled. Its backward calculation remains
        unimplemented.
        """
        if not tensors or not all(isinstance(t, Tensor) for t in tensors):
            raise TypeError("stack requires a nonempty sequence of Tensors.")
        return Stack.apply(*tensors, axis=axis)

    @property
    def shape(self) -> tuple[int, ...]:
        """Sizes of the backing storage's dimensions."""
        return self._backend.shape(self._storage)


    @property
    def ndim(self) -> int:
        """Number of dimensions; a scalar has zero."""
        return len(self.shape)

    @property
    def dtype(self) -> str:
        """storage dtype."""
        return self._backend.dtype(self._storage)

    @property
    def device(self) -> str:
        """Storage device."""
        return self._device

    @property
    def strides(self) -> tuple[int, ...]:
        """Backing array strides in bytes, following NumPy's units."""
        return self._backend.strides(self._storage)

    @property
    def requires_grad(self) -> bool:
        """Whether to track operations on this tensor for autograd."""
        return self._requires_grad

    def numel(self) -> int:
        """Number of logical elements; one for a scalar, zero for an empty array."""
        return self._backend.numel(self._storage)

    def is_contiguous(self) -> bool:
        """Whether the backing array is C-contiguous, including singleton cases."""
        return self._backend.is_contiguous(self._storage)

    def numpy(self) -> np.ndarray:
        """Export an independent NumPy copy; editing it must not change this tensor."""
        return self._backend.to_numpy(self._storage, copy=True)


    # 2. Arithmetic: delegate execution and history recording to Function.apply.

    def __add__(self, other) -> Tensor:
        """Elementwise addition with another Tensor or numeric scalar; allow broadcasting."""
        return Add.apply(self, other)

    def __radd__(self, other) -> Tensor:
        """Addition when this Tensor is the right operand, as in 2 + tensor."""
        return Add.apply(other, self)

    def __sub__(self, other) -> Tensor:
        """Elementwise subtraction with a Tensor or numeric scalar."""
        return Sub.apply(self, other)

    def __rsub__(self, other) -> Tensor:
        """Subtraction when this Tensor is the right operand, as in 2 - tensor."""
        return Sub.apply(other, self)

    def __mul__(self, other) -> Tensor:
        """Elementwise multiplication with a Tensor or numeric scalar."""
        return Mul.apply(self, other)

    def __rmul__(self, other) -> Tensor:
        """Multiplication when this Tensor is the right operand."""
        return Mul.apply(other, self)

    def __truediv__(self, other) -> Tensor:
        """Elementwise division with a Tensor or numeric scalar."""
        return Div.apply(self, other)

    def __rtruediv__(self, other) -> Tensor:
        """Division when this Tensor is the right operand, as in 2 / tensor."""
        return Div.apply(other, self)

    def __matmul__(self, other: Tensor) -> Tensor:
        """Matrix multiplication with another Tensor, following NumPy's shape rules."""
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type(s) for @: 'Tensor' and '{type(other).__name__}'")
        return MatMul.apply(self, other)

    # 3. Views and copying

    def __getitem__(self, key) -> Tensor:
        """Index values; preserve basic slice views and advanced-indexing copies.

        An element selection returns a zero-dimensional Tensor.
        Public element assignment is not supported at this stage.
        """
        return GetItem.apply(self, key=key)

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        """Swap two valid axes, including negative axes, while sharing storage."""
        return Transpose.apply(self, dim0=dim0, dim1=dim1)

    def permute(self, *axes) -> Tensor:
        """Reorder all axes while sharing storage; accept separate axes or a tuple."""
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return Permute.apply(self, axes=axes)

    def reshape(self, *shape) -> Tensor:
        """Change shape in logical row order; share when possible, copy when needed.

        Accept separate dimensions or one tuple. Allow one inferred -1.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if any(dim < -1 for dim in shape):
            raise ValueError("Only -1 may be used as an inferred dimension.")
        return Reshape.apply(self, shape=shape)

    def view(self, *shape) -> Tensor:
        """Change shape without copying element storage, or raise an error.

        Support compatible noncontiguous layouts too. Empty views contain no
        overlapping bytes, so memory-overlap checks alone cannot classify them.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if any(dim < -1 for dim in shape):
            raise ValueError("Only -1 may be used as an inferred dimension.")
        return View.apply(self, shape=shape)

    def expand(self, *shape) -> Tensor:
        """Broadcast to explicit nonnegative dimensions using a shared, read-only view.

        Accept separate dimensions or one tuple. Reject incompatible shapes.
        In this first API, -1 is reserved for reshape/view, not expand.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Expand.apply(self, shape=shape)

    def clone(self) -> Tensor:
        """Copy values into independent storage; this does not promise a particular layout."""
        return Clone.apply(self)

    def contiguous(self) -> Tensor:
        """Return self if already C-contiguous; otherwise wrap a C-contiguous copy."""
        if self.is_contiguous():
            return self
        return Contiguous.apply(self)

    # 4. Reductions

    def sum(self, axis=None, keepdims: bool = False) -> Tensor:
        """Sum over selected axes, or all values when axis is None."""
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims: bool = False) -> Tensor:
        """Mean over selected axes, optionally keeping reduced axes at size one."""
        return Mean.apply(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims: bool = False) -> Tensor:
        """Maximum values over selected axes; return values without an index tensor."""
        return Max.apply(self, axis=axis, keepdims=keepdims)
