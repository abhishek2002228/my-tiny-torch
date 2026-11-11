# Tensor API

The current tinytorch backend provides eager CPU tensors backed by NumPy (`float32` only). Import the public class with `from tinytorch import Tensor`.

## Quick start

```python
from tinytorch import Tensor

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x * x).sum()
loss.backward()
print(x.grad.numpy())  # [2. 4. 6.]
```

`backward()` requires a scalar Tensor with shape `()`. A one-element vector with shape `(1,)` is not a scalar.

## Creating tensors

```python
Tensor(data, requires_grad=False, dtype="float32", device="cpu")
```

The constructor converts input values to float32 and copies them into independent, C-contiguous storage. It accepts scalars, rectangular lists or tuples, NumPy arrays, and other Tensors. NumPy performs value conversion; invalid or ragged input raises an error. Only `device="cpu"` and `dtype="float32"` are supported currently.

Construction starts new gradient history. For example, `Tensor(x, requires_grad=True)` copies `x`'s values and creates a new leaf; backward through the new tensor does not reach `x`.

A nonempty list of equally shaped Tensors can be passed to the constructor to stack **values** along a new axis. This also starts new history. Use `Tensor.stack([a, b], axis=0)` when gradients must flow back to `a` and `b`.

## Metadata and export

| API | Result |
| --- | --- |
| `shape` | Tuple of dimension sizes; a scalar has `()` |
| `ndim` | Number of dimensions |
| `dtype`, `device` | Currently `"float32"` and `"cpu"` |
| `strides` | Byte strides of the current storage, including zero or negative strides |
| `numel()` | Number of logical elements |
| `is_contiguous()` | Whether values are C-contiguous |
| `nbytes` | Logical element bytes (`numel() * 4` for float32) |
| `memory_info()` | Logical bytes, backing allocation bytes when known, and whether storage is a view |
| `numpy()` | Independent, C-contiguous NumPy copy |

`numpy()` is an export of values. Its strides and storage ownership can differ from those of the Tensor. Editing the exported array does not change the Tensor.

## Operations

| API | Behavior and storage |
| --- | --- |
| `+`, `-`, `*`, `/` | Elementwise arithmetic; NumPy broadcasting; new result storage |
| `@` | Matrix multiplication, including vectors and batches; new result storage |
| `x[key]` | Basic slicing shares storage; advanced indexing copies |
| `transpose(dim0, dim1)`, `permute(*axes)` | Reorder axes as shared views |
| `view(*shape)` | Change shape without copying, or raise if impossible |
| `reshape(*shape)` | Share storage when possible; otherwise copy |
| `expand(*shape)` | Read-only broadcast view; expanded axes may have stride zero |
| `clone()` | Independent copy; gradient history remains connected |
| `contiguous()` | Return the same Tensor if already C-contiguous; otherwise copy |
| `Tensor.stack(tensors, axis=0)` | New storage; gradient history remains connected |
| `sum(axis=None, keepdims=False)` | Sum selected axes |
| `mean(axis=None, keepdims=False)` | Average selected axes |
| `max(axis=None, keepdims=False)` | Maximum values over selected axes |

`axis` may be an integer or a tuple. `axis=None` reduces all axes. With `keepdims=True`, reduced axes remain with size one.

### Scalar operands

Python `int`, `float`, and `bool` operands are converted to the Tensor's dtype before arithmetic. Finite values outside the float32 range raise `OverflowError`; ordinary float32 rounding is allowed. Explicit NumPy scalars must already have the Tensor's dtype: `np.float32(2)` is accepted and `np.float64(2)` is rejected. Two Tensor operands currently both have float32 storage.

`tinytorch.inf` is a Python infinity constant. Use `Tensor(inf)` for an infinity tensor or `-inf` for negative infinity.

### Views and strides

```python
x = Tensor([[1, 2, 3], [4, 5, 6]])
print(x.shape, x.strides)                 # (2, 3) (12, 4)
print(x[1:, :2].shape, x[1:, :2].strides) # (1, 2) (12, 4)
print(x.transpose(0, 1).strides)          # (4, 12)
```

The initial storage contains six four-byte values in row order. A slice changes the visible region but keeps the distance between original row starts. A transpose swaps axis sizes and strides; it does not move values.

```python
bias = Tensor([10, 20, 30])
expanded = bias.expand(2, 3)
print(expanded.strides)  # (0, 4)
```

The expanded tensor displays two rows, but it does not store six values. Its zero row stride means each row reads the same three values from `bias`. By contrast, `x + bias` uses broadcasting to add those values to each row of `x` and stores the sums in a new tensor. A view keeps its shared storage alive even if the Tensor it came from is deleted. This API does not support assigning to tensor elements or changing a tensor in place.

## Automatic differentiation

Set `requires_grad=True` on inputs whose gradients you need. Operations on such inputs record their dependencies. `backward()` traverses the graph from a scalar output and accumulates gradients into leaf tensors.

```python
x = Tensor([1, 2, 3], requires_grad=True)
bias = Tensor(2, requires_grad=True)
loss = (x + bias).sum()
loss.backward()
print(x.grad.numpy())     # [1. 1. 1.]
print(bias.grad.numpy())  # 3.0
```

The scalar bias was used three times, so backward sums its three contributions. Each gradient has the same shape as its Tensor. Backward rules for arithmetic, matmul, indexing, views, stack, and reductions are implemented. A backward rule that returns a gradient with the wrong input shape raises an error.

Gradients accumulate across backward calls. By default, `backward()` releases a computed output's graph after use; calling it again on that output raises an error. Call `backward(retain_graph=True)` if you need a later backward pass through that graph. A new forward calculation builds a new graph.

Only leaf gradients are stored by default. Call `intermediate.retain_grad()` before backward to keep an intermediate Tensor's gradient. `grad` is a Tensor with `requires_grad=False`; higher-order gradients are not recorded. 

`loss.dag_viz()` displays the recorded graph in a notebook. Tensor nodes show their shape, a short value preview, and a gradient preview (`None` if no gradient is stored). Use `loss.backward(retain_graph=True)` to inspect the graph after backward; call `retain_grad()` on an intermediate before backward to show its gradient. Use `dag_viz(max_elements=6)` to show more values per node. Outside IPython, the method returns SVG text.

## Current limits

Execution is eager and uses the CPU NumPy backend. GPU storage, lazy execution, optimizers, layers, activations etc are not available yet. See [TODO](../TODO.md) for planned work.
