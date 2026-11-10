"""Operations shared by Tensor's public methods.

Function.apply selects self.backend using the input device before calling
forward. Each operation delegates numerical work to that backend and can later
provide its own backward calculation. Shape/axis/key settings are keyword
arguments to apply. Every forward returns backend storage without rewrapping it.
"""

from .autograd import Function


class Add(Function):
    def forward(self, lhs, rhs):
        return self.backend.add(lhs, rhs)

    def backward(self, grad_output):
        from .tensor import Tensor

        lhs, rhs = self.inputs

        grad_lhs = (
            self.backend.reduce_broadcasted_grad(grad_output, lhs.shape) if isinstance(lhs, Tensor) else None
        )
        grad_rhs = (
            self.backend.reduce_broadcasted_grad(grad_output, rhs.shape) if isinstance(rhs, Tensor) else None
        )
        return (grad_lhs, grad_rhs)

class Sub(Function):
    def forward(self, lhs, rhs):
        return self.backend.sub(lhs, rhs)

    def backward(self, grad_output):
        from .tensor import Tensor

        lhs, rhs = self.inputs

        grad_lhs = (
            self.backend.reduce_broadcasted_grad(grad_output, lhs.shape)
            if isinstance(lhs, Tensor) else None
        )
        grad_rhs = (
            self.backend.reduce_broadcasted_grad(
                self.backend.mul(grad_output, -1), rhs.shape
            )
            if isinstance(rhs, Tensor) else None
        )
        return (grad_lhs, grad_rhs)


class Mul(Function):
    def forward(self, lhs, rhs):
        return self.backend.mul(lhs, rhs)

    def backward(self, array):
        from .tensor import Tensor

        lhs, rhs = self.inputs
        reference = lhs._storage if isinstance(lhs, Tensor) else rhs._storage
        lhs_storage = lhs._storage if isinstance(lhs, Tensor) else self.backend.scalar_like(lhs, reference)
        rhs_storage = rhs._storage if isinstance(rhs, Tensor) else self.backend.scalar_like(rhs, reference)

        grad_lhs = (
            self.backend.reduce_broadcasted_grad(self.backend.mul(array, rhs_storage), lhs.shape)
            if isinstance(lhs, Tensor) else None
        )
        grad_rhs = (
            self.backend.reduce_broadcasted_grad(self.backend.mul(array, lhs_storage), rhs.shape)
            if isinstance(rhs, Tensor) else None
        )
        return (grad_lhs, grad_rhs)

class Div(Function):
    def forward(self, lhs, rhs):
        return self.backend.div(lhs, rhs)

    def backward(self, grad_output):
        from .tensor import Tensor

        lhs, rhs = self.inputs
        reference = lhs._storage if isinstance(lhs, Tensor) else rhs._storage
        left = lhs._storage if isinstance(lhs, Tensor) else self.backend.scalar_like(lhs, reference)
        right = rhs._storage if isinstance(rhs, Tensor) else self.backend.scalar_like(rhs, reference)

        grad_lhs = (
            self.backend.reduce_broadcasted_grad(
                self.backend.div(grad_output, right), lhs.shape
            )
            if isinstance(lhs, Tensor) and lhs.requires_grad else None
        )
        grad_rhs = (
            self.backend.reduce_broadcasted_grad(
                self.backend.div(
                    self.backend.mul(grad_output, self.backend.mul(left, -1)),
                    self.backend.mul(right, right),
                ),
                rhs.shape,
            )
            if isinstance(rhs, Tensor) and rhs.requires_grad else None
        )
        return (grad_lhs, grad_rhs)


class MatMul(Function):
    def forward(self, lhs, rhs):
        return self.backend.matmul(lhs, rhs)

    def backward(self, grad_output):
        lhs, rhs = self.inputs
        grad_lhs, grad_rhs = self.backend.matmul_grad(
            lhs._storage, rhs._storage, grad_output,
            need_lhs=lhs.requires_grad, need_rhs=rhs.requires_grad,
        )
        return (grad_lhs, grad_rhs)


class Stack(Function):
    def forward(self, *arrays):
        return self.backend.stack(*arrays, axis=getattr(self, "axis", 0))

    def backward(self, grad_output):
        return self.backend.unstack(grad_output, axis=getattr(self, "axis", 0))


class GetItem(Function):
    def forward(self, array):
        return self.backend.getitem(array, key=self.key)

    def backward(self, grad_output):
        source = self.inputs[0]
        return (
            self.backend.scatter_add(
                source.shape, self.key, grad_output, dtype=source.dtype
            ),
        )


class Transpose(Function):
    def forward(self, array):
        return self.backend.transpose(array, dim0=self.dim0, dim1=self.dim1)

    def backward(self, grad_output):
        return (
            self.backend.transpose(
                grad_output, dim0=self.dim0, dim1=self.dim1
            ),
        )


class Permute(Function):
    def forward(self, array):
        return self.backend.permute(array, axes=self.axes)

    def backward(self, grad_output):
        ndim = self.inputs[0].ndim
        normalized = tuple(axis % ndim for axis in self.axes)
        inverse = tuple(normalized.index(axis) for axis in range(ndim))
        return (self.backend.permute(grad_output, axes=inverse),)


class Reshape(Function):
    def forward(self, array):
        return self.backend.reshape(array, shape=self.shape)

    def backward(self, grad_output):
        return (self.backend.reshape(grad_output, shape=self.inputs[0].shape),)


class View(Function):
    def forward(self, array):
        return self.backend.view(array, shape=self.shape)

    def backward(self, grad_output):
        return (self.backend.reshape(grad_output, shape=self.inputs[0].shape),)


class Expand(Function):
    def forward(self, array):
        return self.backend.expand(array, shape=self.shape)

    def backward(self, grad_output):
        return (
            self.backend.reduce_broadcasted_grad(
                grad_output, self.inputs[0].shape
            ),
        )


class Clone(Function):
    def forward(self, array):
        return self.backend.clone(array)

    def backward(self, grad_output):
        return (grad_output,)


class Contiguous(Function):
    def forward(self, array):
        return self.backend.contiguous(array)

    def backward(self, grad_output):
        return (grad_output,)


class Sum(Function):
    def forward(self, array):
        return self.backend.sum(array, axis=getattr(self, "axis", None), keepdims=getattr(self, "keepdims", False))

    def backward(self, grad_output):
        input_array = self.inputs[0]._storage
        return (
            self.backend.broadcast_reduced_grad(
                grad_output, input_array.shape,
                axis=getattr(self, "axis", None),
                keepdims=getattr(self, "keepdims", False),
            ),
        )

class Mean(Function):
    def forward(self, array):
        return self.backend.mean(array, axis=getattr(self, "axis", None), keepdims=getattr(self, "keepdims", False))

    def backward(self, grad_output):
        source = self.inputs[0]
        axis = getattr(self, "axis", None)
        count = self.backend.reduction_count(source.shape, axis=axis)
        expanded = self.backend.broadcast_reduced_grad(
            grad_output, source.shape,
            axis=axis, keepdims=getattr(self, "keepdims", False),
        )
        return (self.backend.div(expanded, count),)


class Max(Function):
    def forward(self, array):
        result = self.backend.max(
            array, axis=getattr(self, "axis", None),
            keepdims=getattr(self, "keepdims", False),
        )
        self._maxima = result # to avoid backward to recompute maxima
        return result

    def backward(self, grad_output):
        return (
            self.backend.max_grad(
                self.inputs[0]._storage, self._maxima, grad_output,
                axis=getattr(self, "axis", None),
                keepdims=getattr(self, "keepdims", False),
            ),
        )
