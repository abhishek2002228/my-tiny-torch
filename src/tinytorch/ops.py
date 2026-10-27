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


class Sub(Function):
    def forward(self, lhs, rhs):
        return self.backend.sub(lhs, rhs)


class Mul(Function):
    def forward(self, lhs, rhs):
        return self.backend.mul(lhs, rhs)


class Div(Function):
    def forward(self, lhs, rhs):
        return self.backend.div(lhs, rhs)


class MatMul(Function):
    def forward(self, lhs, rhs):
        return self.backend.matmul(lhs, rhs)


class Stack(Function):
    def forward(self, *arrays):
        return self.backend.stack(*arrays, axis=getattr(self, "axis", 0))


class GetItem(Function):
    def forward(self, array):
        return self.backend.getitem(array, key=self.key)


class Transpose(Function):
    def forward(self, array):
        return self.backend.transpose(array, dim0=self.dim0, dim1=self.dim1)


class Permute(Function):
    def forward(self, array):
        return self.backend.permute(array, axes=self.axes)


class Reshape(Function):
    def forward(self, array):
        return self.backend.reshape(array, shape=self.shape)


class View(Function):
    def forward(self, array):
        return self.backend.view(array, shape=self.shape)


class Expand(Function):
    def forward(self, array):
        return self.backend.expand(array, shape=self.shape)


class Clone(Function):
    def forward(self, array):
        return self.backend.clone(array)


class Contiguous(Function):
    def forward(self, array):
        return self.backend.contiguous(array)


class Sum(Function):
    def forward(self, array):
        return self.backend.sum(array, axis=getattr(self, "axis", None), keepdims=getattr(self, "keepdims", False))


class Mean(Function):
    def forward(self, array):
        return self.backend.mean(array, axis=getattr(self, "axis", None), keepdims=getattr(self, "keepdims", False))


class Max(Function):
    def forward(self, array):
        return self.backend.max(array, axis=getattr(self, "axis", None), keepdims=getattr(self, "keepdims", False))
