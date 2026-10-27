"""Run after installing the package: python examples/tensor_basics.py."""

from tinytorch import Tensor


def main():
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    bias = Tensor([10, 20, 30])
    output = x + bias
    print("Broadcast addition:\n", output.numpy())
    print("Transpose:\n", output.transpose(0, 1).numpy())
    print("Row means:\n", output.mean(axis=1, keepdims=True).numpy())
    print("Centered rows:\n", (output - output.mean(axis=1, keepdims=True)).numpy())
    print("Flat view:", x.view(6).numpy())
    print("Stack:\n", Tensor.stack([bias, bias]).numpy())
    print("Metadata:", x.shape, x.dtype, x.device, x.strides)


if __name__ == "__main__":
    main()
