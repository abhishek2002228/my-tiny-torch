# My TinyTorch

A Python tensor library built from scratch, inspired by [TinyTorch](https://mlsysbook.ai/tinytorch).

The current API supports eager CPU float32 tensors, arithmetic, matrix multiplication,
indexing, shared views, copies, stacking, and reductions. Operations can record their
input connections, but **backward calculations and GPU execution are not implemented**.

## Quick setup for notebooks and development

From the downloaded or cloned repository, run with Python 3.10 or later:

```bash
python3 setup_env.py
```

This creates `.venv` if needed, installs the library in editable mode with test,
build, and notebook-kernel dependencies, and registers a kernel inside that
environment. An internet connection is needed to download missing dependencies.
You can rerun it to reuse the environment. It does not activate your terminal.

Open `examples/my_tiny_torch_playground.ipynb`. In VS Code's kernel picker select
**Python Environments → this repository's `.venv/bin/python`** (on Windows,
`.venv\\Scripts\\python.exe`). The setup script prints the exact path.
The kernel may also appear as **My Tiny Torch (.venv)**.

## Install and use

Requires Python 3.10 or later. From a clone of this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Or install directly from GitHub into your chosen environment:

```bash
python -m pip install "git+https://github.com/abhishek2002228/my-tiny-torch.git"
```

The distribution is named `my-tiny-torch`; the Python import is `tinytorch`.
Use an environment without the original TinyTorch tutorial package, which has
that same import name. Once installed, imports work outside the repository too:

```python
from tinytorch import Tensor

x = Tensor([[1, 2, 3], [4, 5, 6]])
bias = Tensor([10, 20, 30])
y = x + bias

print(y.numpy())                  # [[11. 22. 33.], [14. 25. 36.]]
print(y.mean(axis=1).numpy())     # [22. 25.]
print(x.transpose(0, 1).numpy())  # [[1. 4.], [2. 5.], [3. 6.]]
print(x.shape, x.dtype, x.device) # (2, 3) float32 cpu
```

Run a complete example with `python examples/tensor_basics.py` after installation.
The GitHub install command installs the version committed and pushed there.

## Tensor API

| API | Behavior |
| --- | --- |
| `Tensor(data, dtype="float32", device="cpu", requires_grad=False)` | Copy and convert input into independent contiguous storage; start fresh history |
| `shape`, `ndim`, `dtype`, `device`, `strides` | Read metadata; dtype is a string and strides are in bytes |
| `numel()`, `is_contiguous()` | Inspect logical size and row-major contiguity |
| `+`, `-`, `*`, `/` | Elementwise Tensor/scalar arithmetic with NumPy broadcasting |
| `@` | Tensor matrix multiplication, including batches and vector dot products |
| `x[key]` | NumPy indexing: basic slices share storage, advanced indexing copies |
| `transpose(a, b)`, `permute(*axes)` | Reorder axes while sharing storage |
| `view(*shape)` | Change shape without copying or raise |
| `reshape(*shape)` | Share storage if possible; copy if necessary |
| `expand(*shape)` | Broadcast a shared read-only view |
| `clone()` | Copy values into independent storage |
| `contiguous()` | Return self if contiguous, otherwise pack a row-major copy |
| `Tensor.stack(tensors, axis=0)` | Stack into new storage, retaining operation history when requested |
| `sum`, `mean`, `max` | Reduce with `axis=None` and `keepdims=False` defaults; max returns values |
| `numpy()` | Export an independent NumPy copy |

## Development

Use an editable install so edits under `src` are picked up by new Python sessions:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m build
```

For notebooks, also install the optional kernel dependency:

```bash
python -m pip install -e '.[dev,notebook]'
```

In VS Code select **this repository's `.venv/bin/python`** as the notebook kernel.
Restart the kernel after changing imported library code. No `sys.path` edits are needed.

### Verification

- `tests/test_tensor.py`: public results, shapes, errors, construction, and storage contracts.
- `tests/test_ops.py`: direct operations, scalar normalization, and view behavior.
- `tests/test_backends.py`: backend selection and dispatch without extra copying.
- `tests/test_numpy_backend.py`: CPU creation, validation, metadata, and exports.
- `tests/test_autograd.py`: public API dispatch and graph connections, not derivative correctness.

GitHub Actions runs the suite, builds a source distribution and wheel, and installs
that wheel into a separate environment to run the public example outside the checkout.

## Explore

- [Tensor basics and contracts](docs/tensors.md)
- [My TinyTorch playground](examples/my_tiny_torch_playground.ipynb)
