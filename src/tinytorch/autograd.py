"""Shared execution and gradient-history setup for eager tensor operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from abc import ABC, abstractmethod

from .backends import get_backend

if TYPE_CHECKING:
    from .tensor import Tensor


class Function(ABC):
    """Base class for ops
    Subclasses implement forward using a backend selected from input devices.
    Each call to apply creates
    a separate operation instance, so forward can save values or shapes on self
    for that particular call's backward calculation.

    This class records gradient connections; it does not yet traverse the graph
    or accumulate gradients. Tensor.backward will handle that later.
    """

    def __init__(self, *inputs, **params):
        self.inputs = inputs
        for k, v in params.items():
            setattr(self, k, v)

    @abstractmethod
    def forward(self, *operands):
        pass

    def backward(self, grad_output):
        raise NotImplementedError(
            f"{type(self).__name__}.backward is not implemented."
        )

    @classmethod
    def apply(cls, *inputs, **params):
        """ Run this operation on inputs, saving any context needed for backward.
            Returns a new Tensor.
        """
        from .tensor import Tensor
        op = cls(*inputs, **params)

        tensor_inputs = [value for value in inputs if isinstance(value, Tensor)]

        if not tensor_inputs:
            raise ValueError("At least one input must be a Tensor.")

        #ensure all input Tensors are of the same device type
        if any(value.device != tensor_inputs[0].device for value in tensor_inputs):
            raise ValueError("All input Tensors must be on the same device.")

        op.device = tensor_inputs[0].device
        op.backend = get_backend(op.device)

        operands = []
        for value in inputs:
            if isinstance(value, Tensor):
                operands.append(value._storage)
            elif isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
                operands.append(value)
            else:
                raise TypeError(f"Unsupported input type: {type(value)}")

        result = op.forward(*operands)

        # does my output tensor need requires grad? It does if any of the inputs require grad.
        requires_grad = any(value.requires_grad for value in tensor_inputs)

        if not isinstance(result, op.backend.storage_type):
            raise TypeError(
                f"Forward must return a {op.backend.storage_type.__name__}, got {type(result)}"
            )
        # result here can refer to the same backing array or a new one,
        # depending on the operation. We wrap it up in a new Tensor and return it.
        output = Tensor._from_storage(result, device=op.device, requires_grad=requires_grad)

        if requires_grad:
            output._ctx = op

        return output
