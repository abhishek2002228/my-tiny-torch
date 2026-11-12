# Readings 
### Autograd -  memory optimization 
- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
  Can we train deep nets in sublinear memory for a constant compute overhead?
  Checkpoint every √N layers and recompute the rest: O(√N) memory for ~one extra forward pass.
- [Dynamic Tensor Rematerialization](https://arxiv.org/abs/2006.09616)
  Can store-vs-recompute be decided at runtime, without a static graph (i.e. in an eager framework)?
  Evicts tensors greedily by a cost/(size × staleness) heuristic and gets close to the optimal plan.
- [Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization](https://arxiv.org/pdf/1910.02653)
  What is the *optimal* store/recompute plan on real graphs with non-uniform op costs, and how far off is √N?
  Formulates rematerialization as an integer linear program under a memory budget.
- [MONeT: Memory Optimization for Deep Networks](https://openreview.net/pdf?id=bnY0jm4l59)
  Should checkpointing be chosen jointly with *how* each op is implemented (memory-hungry fast kernel vs lean slow one)?
  Optimizes checkpoint schedule and per-op implementation together, beating checkpointing alone.
## My resources 
- [DL Systems course](https://dlsyscourse.org)
- [Language Modeling from Scratch](https://cs336.stanford.edu)
- [TinyGrad](https://github.com/tinygrad/tinygrad)