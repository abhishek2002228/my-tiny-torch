
## Next steps

- verify autograd
- Optimize tinytorch
    - optimize the numpy backend allocations to improve memory usage. 
- Add activation functions and neural network layers with working backward rules.
- Add loss functions, optimizers so parameters can be trained.
- Add datasets and a DataLoader for batching training data.
- Train a minimal language model end to end with this framework.
- Build a memory/latency profiler for tinytorch
- Design the GPU backend: memory allocation, transfers, operation dispatch, kernels etc.
- Multi-GPU training/inference library 
- Design lazy execution: decide how to represent operations, when to realize results, and how to fuse operations such as multiply followed by reduction. A minimal compiler ? 
