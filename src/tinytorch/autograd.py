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


def autograd_engine(output: "Tensor", retain_graph=False):
    from .tensor import Tensor
    backend = get_backend(output.device)

    if output.shape != ():
        raise ValueError("backward requires a scalar tensor.")

    if not output.requires_grad:
        raise ValueError("backward requires a tensor with requires_grad=True.")

    # Post-order traversal: append a tensor only after its parents.
    # An explicit stack handles chains longer than Python's recursion limit.
    ordered = []
    visited = set()
    visiting = set()
    stack = [(output, False)]

    while stack:
        tensor, expanded = stack.pop()
        if tensor._graph_released:
            raise RuntimeError("Cannot backward through a released graph.")
        key = id(tensor)
        if expanded:
            visiting.remove(key)
            visited.add(key)
            ordered.append(tensor)
            continue
        if key in visited:
            continue
        if key in visiting:
            raise RuntimeError("Cycle detected in autograd graph.")

        visiting.add(key)
        stack.append((tensor, True))
        if tensor._ctx is not None:
            for value in reversed(tensor._ctx.inputs):
                if isinstance(value, Tensor) and value.requires_grad:
                    stack.append((value, False))

    ordered.reverse()

    # Gradients for this backward call only. 
    pending = {id(output): backend.ones_like(output._storage)}
    
    # chain rule, accumulate gradients
    for tensor in ordered:
        grad = pending.pop(id(tensor), None)
        if grad is None:
            continue

        # Leaf tensors and tensors with retained gradients store their own gradient.
        if tensor._ctx is None or tensor._retain_grad:
            if tensor._grad is None:
                tensor._grad = Tensor._from_storage(grad, device=tensor.device, requires_grad=False)
            else:
                accumulated = backend.add(tensor._grad._storage, grad)
                tensor._grad = Tensor._from_storage(accumulated, device=tensor.device, requires_grad=False)

        # A leaf has no backward
        if tensor._ctx is None:
            continue

        input_grads = tensor._ctx.backward(grad)

        if len(input_grads) != len(tensor._ctx.inputs):
            raise RuntimeError("backward must return one gradient per input.")

        for value, contribution in zip(tensor._ctx.inputs, input_grads):
            if isinstance(value, Tensor) and value.requires_grad and contribution is not None:
                actual_shape = getattr(contribution, "shape", None)
                if actual_shape != value.shape:
                    raise ValueError(
                        f"{type(tensor._ctx).__name__}.backward returned gradient shape "
                        f"{actual_shape} for input shape {value.shape}."
                    )
                key = id(value)
                if key in pending:
                    pending[key] = backend.add(pending[key], contribution)
                else:
                    pending[key] = contribution
        if not retain_graph:
            tensor._ctx = None # can delete the context for this tensor once grads are used.
            # Without this, memory grows with each training step. 
            tensor._graph_released = True

                    
                
def dag_viz(output: "Tensor") -> str | None:
    """Display the recorded autograd DAG in a notebook and return its SVG.

    Call before backward releases the graph, or use retain_graph=True.
    Only operations whose outputs require gradients have recorded history.
    """
    from html import escape
    from .tensor import Tensor

    if not isinstance(output, Tensor):
        raise TypeError("dag_viz expects a Tensor.")
    if output._graph_released:
        raise RuntimeError("This graph was released; visualize before backward or retain it.")

    # Visit parents before their outputs, without relying on Python recursion.
    ordered = []
    visited = set()
    visiting = set()
    stack = [(output, False)]
    while stack:
        tensor, expanded = stack.pop()
        if tensor._graph_released:
            raise RuntimeError("This graph was released; visualize before backward or retain it.")
        key = id(tensor)
        if expanded:
            visiting.remove(key)
            visited.add(key)
            ordered.append(tensor)
            continue
        if key in visited:
            continue
        if key in visiting:
            raise RuntimeError("Cycle detected in autograd graph.")
        visiting.add(key)
        stack.append((tensor, True))
        if tensor._ctx is not None:
            for value in reversed(tensor._ctx.inputs):
                if isinstance(value, Tensor):
                    stack.append((value, False))

    nodes = []
    edges = []
    seen = {}
    for tensor in ordered:
        parents = []
        context = tensor._ctx
        if context is not None:
            for value in context.inputs:
                if isinstance(value, Tensor):
                    parents.append(seen[id(value)])
                else:
                    constant_id = f"c{len(nodes)}"
                    nodes.append((constant_id, repr(value), 0, "constant"))
                    parents.append((constant_id, 0))

            operation_depth = max((depth for _, depth in parents), default=0) + 1
            operation_id = f"o{len(nodes)}"
            nodes.append((operation_id, type(context).__name__, operation_depth, "operation"))
            edges.extend((parent_id, operation_id) for parent_id, _ in parents)
            tensor_depth = operation_depth + 1
        else:
            tensor_depth = 0

        tensor_id = f"t{len(seen)}"
        label = f"T{len(seen)}  {tensor.shape}"
        kind = "output" if tensor is output else "tensor"
        nodes.append((tensor_id, label, tensor_depth, kind))
        seen[id(tensor)] = (tensor_id, tensor_depth)
        if context is not None:
            edges.append((operation_id, tensor_id))

    columns = {}
    for node_id, label, depth, kind in nodes:
        columns.setdefault(depth, []).append((node_id, label, kind))
    width = 190 * len(columns) + 40
    height = 75 * max(map(len, columns.values())) + 40
    positions = {}
    for depth, column in columns.items():
        for row, (node_id, _, _) in enumerate(column):
            positions[node_id] = (20 + 190 * depth, 20 + 75 * row)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#64748b"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for source, target in edges:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        start_x, end_x = x1 + 145, x2
        middle = (start_x + end_x) / 2
        parts.append(
            f'<path d="M{start_x},{y1 + 20} C{middle},{y1 + 20} {middle},{y2 + 20} {end_x - 5},{y2 + 20}" '
            'fill="none" stroke="#64748b" stroke-width="1.5" marker-end="url(#arrow)"/>'
        )
    colors = {"tensor": "#e0f2fe", "output": "#bbf7d0", "operation": "#ffedd5", "constant": "#f1f5f9"}
    for node_id, label, _, kind in nodes:
        x, y = positions[node_id]
        parts.append(f'<rect x="{x}" y="{y}" width="145" height="40" rx="7" fill="{colors[kind]}" stroke="#334155"/>')
        parts.append(f'<text x="{x + 72.5}" y="{y + 25}" text-anchor="middle" font-family="monospace" font-size="13" fill="#0f172a">{escape(label)}</text>')
    parts.append('</svg>')
    svg = ''.join(parts)

    try:
        from IPython import get_ipython
        from IPython.display import SVG, display
    except ImportError:
        pass
    else:
        if get_ipython() is not None:
            display(SVG(data=svg))
            return None
    return svg
