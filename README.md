# TinyTorch

A small tensor library inspired by [TinyGrad](https://github.com/tinygrad/tinygrad). It currently runs eagerly on the CPU with NumPy. Tensor API and autograd are implemented. GPU execution, lazy execution, and training components are planned.

## Install

Requires Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

You can also install the published repository into another environment with `python -m pip install "git+https://github.com/abhishek2002228/my-tiny-torch.git"`.

## Use

```python
from tinytorch import Tensor

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x * x).sum()
loss.backward()

print(loss.numpy())   # 14.0
print(x.grad.numpy()) # [2. 4. 6.]
```

`backward()` takes a scalar Tensor. It stores gradients on leaf inputs and releases the graph by default. See the [Tensor API](docs/tensors.md) for operations, storage sharing, broadcasting, and gradient rules.

Run `python examples/tensor_basics.py` for another example.

## Notebooks

```bash
python3 setup_env.py
```

This creates or reuses `.venv`, installs editable library and notebook dependencies, and registers the **My Tiny Torch (.venv)** Jupyter kernel. Open [my_tiny_torch_playground.ipynb](examples/my_tiny_torch_playground.ipynb) and select that kernel if prompted. Restart the kernel after changing library code. Rerun setup after moving the repository or recreating `.venv`.

## Test

Run tests in the main `.venv` environment created by `python3 setup_env.py`:

```bash
.venv/bin/python -m pytest -q
```

## Project status

See [TODO.md](TODO.md) for the path toward layers, training a small language model, a GPU backend, and lazy execution.
