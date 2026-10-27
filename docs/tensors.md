# Tensor basics and contracts

A tensor is a multidimensional array of numerical values organized along a
defined sequence of coordinate axes.

The CPU tensor API described below is implemented. Gradient connections are recorded, but backward calculations, detach, and mutation tracking remain future work; their intended contracts are described separately below.

**Copy user input when constructing a tensor. Wrap internal operation results without copying them again.**

## What a tensor represents

A tensor combines numerical storage with metadata describing how to read it. The first backend uses NumPy arrays on the CPU, with `float32` values. Other dtypes and devices can be added explicitly later.

| Property | Meaning |
| --- | --- |
| `shape` | Number of elements along each dimension |
| `ndim` | Number of dimensions: `len(shape)` |
| `numel()` | Total number of logical elements: product of the shape |
| `dtype` | String naming the representation of each element; initially `"float32"` |
| `device` | Location of the storage; initially CPU |
| `strides` | Bytes to move along each dimension, following NumPy's units |
| Offset, conceptually | Distance from the backing allocation's start to the view's first element |

A scalar has shape `()`, zero dimensions, and one element. Empty tensors have zero elements. Metadata must describe the current backing array; it cannot be changed independently of that array. PyTorch reports strides in elements, so its stride numbers differ from our byte strides.

A new Tensor object does not necessarily mean a new allocation for its values. Multiple tensors can refer to the same storage with different shapes, strides, and starting positions.

## Construction and internal wrapping

`Tensor(data, dtype="float32")` creates independent, contiguous storage. Another Tensor and nonempty lists or tuples of equally shaped Tensors are handled by the Tensor constructor. Ordinary input is passed to the backend's `create` method. Subsequent changes to an input array cannot affect the tensor.

**Construction delegates input conversion to NumPy.** The CPU backend calls `np.array(data, dtype=numpy_dtype, copy=True, order="C")` directly. It does not first call `np.asarray`, inspect the first element, or recursively validate values. Numeric strings such as `["1", "2.5"]`, mixed numeric/text input such as `[1, "2"]`, and convertible object arrays are accepted. Booleans become `1.0` and `0.0` in float32 storage. Input that NumPy cannot convert, such as `["hello"]` or ragged nested lists, raises NumPy's own error. Other conversion behavior, including warnings or lossy conversions, follows NumPy rather than a separate framework policy. The backend accepts lists, tuples, ndarrays, and scalars as ordinary input containers. It checks the requested dtype against its supported mapping; CPU float32 is currently supported.

This avoids a preliminary ndarray allocation for list input, but does not guarantee that NumPy internally needs only one allocation or that every input converts faster. Public construction still deliberately copies. Internal operation results use `_from_storage` and are wrapped without copying or dtype conversion.

Constructing from another Tensor copies its values and uses the new constructor's gradient request rather than inheriting the input's history. This permissive conversion policy applies to construction, not arithmetic operands: passing a string directly to an operation remains unsupported.

**Stacking during construction copies values and starts fresh gradient history.** For example:

```python
a = Tensor([1, 2], requires_grad=True)
b = Tensor([3, 4], requires_grad=True)
c = Tensor([a, b], requires_grad=True)
# c contains [[1, 2], [3, 4]] in independent, row-major storage.
```

The constructor combines raw arrays along a new leading axis. It does not record that c came from a and b. When autograd is implemented, c is a new leaf: later calculations can produce c's gradient, but cannot propagate through construction to a or b. Setting `requires_grad=True` requests gradients for c itself; it does not restore those connections. Existing gradients on a and b are not erased or copied. Their shapes must match for stacking. An empty list remains an ordinary empty numeric tensor; it is not a request to stack zero tensors.

**The separate `Tensor.stack([a, b], axis=0)` operation preserves gradient connections.** It creates independent result storage but records its inputs when gradients are tracked. Backward splits the incoming gradient along the new axis and sends each part to its corresponding input. If an input appears more than once, its contributions accumulate. Stack forward execution and input recording are implemented; its backward calculation remains unimplemented.

For example, after constructing `x = Tensor(source)`, changing `source[0]` leaves `x` unchanged. Copying is deliberate here: callers retain control of their input without accidentally changing framework values.

Operations use the private `_from_storage` path for existing backend storage (a NumPy array on CPU). It keeps that array directly, including its layout and any reference to shared storage. It accepts supported arrays and validates them; it does not silently cast their dtype, make them contiguous, or copy them. Operations perform any necessary conversion before wrapping.

The operation determines whether it needs new storage. Wrapping only creates the Tensor object:

```text
Construction: user array -> copy -> Tensor
Addition:     input arrays -> new result array -> wrap -> Tensor
Transpose:    input array -> NumPy view -> wrap -> Tensor
```

This avoids both copying views and copying newly computed results a second time. Zero copy refers to the element storage; creating an array descriptor or Tensor object still has a small cost.

## Backend boundary

Tensor stores `_storage`, `_backend`, and `_device`. The constructor unwraps Tensor
inputs and delegates copying to `backend.create`. Properties delegate metadata to
the backend; `_from_storage` calls `validate_storage` and retains the exact object.
The NumPy backend owns native dtype mapping and exposes names through
`supported_dtypes`. Adding a native dtype mapping extends that backend's capability.

Public operations call `Function.apply`, which checks matching devices, selects a
backend, extracts storage, passes scalar operands unchanged, and calls `forward`.
Each operation delegates numerical work to its backend. Backend methods normalize
scalar numerical results to ndarrays. Unsupported result dtypes are rejected at
wrapping, with no implicit conversion. NumPy 1.x and 2.x have different scalar
promotion rules, so some scalar combinations may be unsupported on one version.
History is attached only when an input requires gradients.

## Worked examples: shape, strides, and offset

These examples describe our intended Tensor behavior. The NumPy examples at the end are runnable now and show the backend mechanisms we plan to preserve.

We use this matrix throughout:

```python
x = Tensor([[1, 2, 3],
            [4, 5, 6]])
```

Public construction copies input into contiguous row-major `float32` storage. The six values occupy 24 bytes, with each four-byte number immediately after the previous one:

```text
Storage index:    0    1    2    3    4    5
Byte offset:     0    4    8   12   16   20
Value:           1    2    3    4    5    6
```

**Shape** says which coordinates exist. **Strides** say how far each coordinate step moves in storage. **Offset** says where the first value of a particular view begins. A view can change these without moving the stored values.

NumPy already keeps the array's starting pointer and references to its backing storage. We do not need a custom allocator or a separate stored offset field for the first backend. The offset is useful for explaining how a view maps coordinates into the original allocation.

### Row-major strides: multiply the sizes to the right

For our default packed row-major construction, advancing one column moves one element. Advancing one row skips three elements:

```text
shape:             (2, 3)
element strides:   (3, 1)
byte strides:      (12, 4)
offset:            0
```

For a nonempty shape `(D0, D1, ..., Dn)`, the usual packed element strides follow:

```text
last stride = 1
stride[k] = stride[k + 1] * shape[k + 1]
```

Equivalently, each stride is the product of the dimension sizes to its right. For shape `(2, 3, 4)`:

```text
element strides: (3*4, 4, 1) = (12, 4, 1)
float32 bytes:                 (48, 16, 4)
```

Moving along the first axis skips a whole 12-element matrix; moving along the second skips a four-element row; moving along the last skips one element. This computes a default packed layout. It is not a rule for recomputing strides whenever a view's shape changes.

### Indexing: add the offset and the coordinate steps

Using element units, the general mapping is:

```text
storage index = offset + i0*stride[0] + i1*stride[1] + ...
```

For our matrix, `x[1, 2]` maps to:

```text
index = 0 + 1*3 + 2*1 = 5
value = 6
```

Using bytes instead gives `0 + 1*12 + 2*4 = 20`, the same location. Do not mix element offsets with byte strides. The indexing operation must first check that coordinates are valid; this arithmetic describes the location of a valid coordinate.

### Slicing: change the visible region, retain the layout

```python
y = x[1:, :2]
```

`1:` selects rows starting at row one; `:2` selects columns zero and one. The result is `[[4, 5]]`:

```text
shape:             (1, 2)
element strides:   (3, 1)
byte strides:      (12, 4)
element offset:    3, relative to x's allocation
byte offset:       12
```

The first selected value now lives at storage index three. The original distance between row starts is still three elements, so the slice inherits that row stride. It does not recalculate the stride as two just because it exposes two columns. In this one-row result, no valid coordinate ever steps along the row axis, so its row stride is unused.

For `y[0, 1]`, the mapping is `3 + 0*3 + 1*1 = 4`, giving value `5`.

Compare a slice containing two rows:

```python
z = x[:, :2]  # [[1, 2], [4, 5]]
```

Its shape is `(2, 2)`, strides are `(3, 1)` elements, and offset is zero. Row starts remain three positions apart. Reading the selected values row by row skips the unselected `3`, so this view is not contiguous in row order.

A slice with a step can change strides too: `x[:, ::2]` selects `[[1, 3], [4, 6]]`, with shape `(2, 2)` and element strides `(3, 2)`. Advancing a selected column now skips two stored elements. Both examples share x's storage in our design.

### Transpose: swap axes and their strides

```python
t = x.transpose(0, 1)
```

```text
Logical values:    [[1, 4],
                    [2, 5],
                    [3, 6]]
shape:             (3, 2)
element strides:   (1, 3)
byte strides:      (4, 12)
offset:            0
```

The backing allocation still stores `[1, 2, 3, 4, 5, 6]`. In t, moving one row steps one element in that allocation; moving one column steps three. For example, `t[2, 1]` maps to `0 + 2*1 + 1*3 = 5`, the same value as `x[1, 2]`.

The transpose's last element stride is three. A last stride of one describes the usual packed row-major layout, not every tensor view. Our internal wrapper must retain this NumPy view instead of copying it through public construction.

### Broadcasting: reuse values with a zero stride

```python
bias = Tensor([10, 20, 30])
expanded = bias.expand(2, 3)
result = x + bias
```

The bias has its own three-element allocation. Expansion gives a logical view over that allocation:

```text
Logical values:    [[10, 20, 30],
                    [10, 20, 30]]
shape:             (2, 3)
element strides:   (0, 1)
byte strides:      (0, 4)
offset:            0, relative to the bias allocation
```

Both `expanded[0, 1]` and `expanded[1, 1]` map to bias index one:

```text
0 + 0*0 + 1*1 = 1
0 + 1*0 + 1*1 = 1
```

The row coordinate has no effect on the address. The view has six logical elements but uses only three stored bias values. There is no allocation for duplicated bias values; there are still small view and Tensor objects.

Addition produces `[[11, 22, 33], [14, 25, 36]]` in independent result storage. Broadcasting saves duplication of the input, not allocation of the answer. NumPy can perform these broadcast reads directly without us explicitly calling expand first. Its explicit `broadcast_to` result is read-only and can map multiple positions to one value; our initial API does not support writes to these views. See [NumPy broadcasting](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html).

Backward must reverse this reuse by summing contributions: if the incoming output gradient is `[[1, 2, 3], [4, 5, 6]]`, the bias gradient is `[5, 7, 9]`, with shape `(3,)`.

### Contiguity: check the traversal order, not ownership

For arrays whose dimensions all exceed one, the usual row-major stride formula tells us whether their values are packed in row traversal order. For t's shape `(3, 2)`, packed strides would be `(2, 1)`; its actual strides `(1, 3)` describe a different traversal order.

```python
packed = t.contiguous()
```

This allocates row-major storage for t's logical values:

```text
New buffer:        [1, 4, 2, 5, 3, 6]
shape:             (3, 2)
element strides:   (2, 1)
byte strides:      (8, 4)
offset:            0 in the new allocation
```

Our `x.contiguous()` returns x itself because x already has the requested layout. The internal wrapper keeps the new array when a copy is needed, without another constructor copy.

Contiguity does not tell us whether a tensor owns its storage. A slice can be contiguous and still share an allocation. In particular, `x[1:, :2]` is C-contiguous even with strides `(3, 1)`: it contains one row, so only the consecutive values `4, 5` are traversed. Size-one dimensions have unused strides, and empty arrays need special handling too. Use NumPy's `array.flags.c_contiguous` as the backend's authoritative check rather than strict equality with a computed stride tuple. See [NumPy array flags](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html).

Neither shape nor strides prove storage sharing. Two independent arrays can have identical metadata. For nonempty NumPy examples, use `np.shares_memory` to check overlapping storage; empty arrays require other ownership checks.

### Reshape: preserve logical order, share or copy as needed

```python
a = x.reshape(3, 2)  # [[1, 2], [3, 4], [5, 6]]
```

The required order `[1, 2, 3, 4, 5, 6]` already matches x's storage. This result can share it, with element strides `(2, 1)`.

Now reshape the transpose:

```python
b = t.reshape(3, 2)  # [[1, 4], [2, 5], [3, 6]]
flat = t.reshape(6) # [1, 4, 2, 5, 3, 6]
```

The first request retains t's existing shape and can share its storage. The flattening request needs a copy: the storage positions for t's logical sequence are `0, 3, 1, 4, 2, 5`, which cannot be expressed with one fixed stride. Reshape does not always make storage contiguous. It preserves logical order and copies when the requested shape cannot use the existing layout. Our strict `t.view(6)` must raise an error instead of copying.

### Reductions: combine values along an axis

A reduction combines multiple inputs into output values. The axis names the dimension being reduced. For our `(2, 3)` matrix:

| Operation | Values | Shape |
| --- | --- | --- |
| `x.sum(axis=0)` | `[5, 7, 9]` | `(3,)` |
| `x.sum(axis=1)` | `[6, 15]` | `(2,)` |
| `x.sum()` | `21` | `()` |
| `x.mean(axis=1)` | `[2, 5]` | `(2,)` |
| `x.max(axis=0)` | `[4, 5, 6]` | `(3,)` |

Axis zero combines rows, leaving one answer per column. Axis one combines columns, leaving one answer per row. By default the reduced axis disappears. NumPy calls the argument `axis`; PyTorch calls it `dim`. Both can retain reduced axes with length one: NumPy uses `keepdims=True`, while PyTorch uses `keepdim=True`.

Keeping the dimension matters when using the result in another broadcast:

```python
row_means = x.mean(axis=1, keepdims=True)  # [[2], [5]], shape (2, 1)
centered = x - row_means                 # [[-1, 0, 1], [-1, 0, 1]]
```

Without keepdims, row means have shape `(2,)`. Broadcasting aligns that two with x's last dimension of three, causing an error. Shape `(2, 1)` correctly reuses one mean across each row's three columns. If the input happened to be square, the wrong shape could instead broadcast along the wrong axis without an error. See [NumPy reduction parameters](https://numpy.org/doc/stable/reference/generated/numpy.sum.html).

Reductions calculate output values and allocate result storage. For ordinary sum, mean, and max reductions over this input, every input element is read. Traversal order affects locality: reading each row visits adjacent values; reading down a column steps three elements at a time. Actual backend implementations may reorder or optimize their loops, so this observation is not a guaranteed timing ratio.

### The same examples in NumPy

This standalone code demonstrates the backend behavior without requiring our Tensor class. Offset is measured in bytes relative to the specified source allocation. The helper uses ordinary nonempty arrays and is not an ownership detector for arbitrary arrays.

```python
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32, order="C", copy=True)
bias = np.array([10, 20, 30], dtype=np.float32)

def describe(name, array, source):
    offset_bytes = array.ctypes.data - source.ctypes.data
    print(name, array.tolist())
    print("shape:", array.shape)
    print("byte strides:", array.strides)
    print("element strides:", tuple(s // array.itemsize for s in array.strides))
    print("offset bytes:", offset_bytes)
    print("shares memory with source:", np.shares_memory(array, source))
    print("C-contiguous:", array.flags.c_contiguous)

describe("original", x, x)
describe("one-row slice", x[1:, :2], x)
describe("two-row slice", x[:, :2], x)
describe("transpose", x.T, x)
describe("expanded bias", np.broadcast_to(bias, (2, 3)), bias)
describe("shared reshape", x.reshape(3, 2), x)

flat = x.T.reshape(6)
packed = np.ascontiguousarray(x.T)
print("Flattened transpose:", flat.tolist(), "shares x:", np.shares_memory(flat, x))
print("Packed transpose strides:", packed.strides, "shares x:", np.shares_memory(packed, x))
print("Column sums:", x.sum(axis=0))
print("Row sums:", x.sum(axis=1))
print("Centered rows:", x - x.mean(axis=1, keepdims=True))
```

For a NumPy view, our operation should wrap that exact array through the private path. For an allocated result, it should also wrap the exact array, avoiding a second copy. Public construction remains the place where we deliberately copy into our default row-major layout.

## Operation contracts

| Operation | Storage contract |
| --- | --- |
| `transpose(dim0, dim1)` / `permute(axes)` | Share storage; change the axis order and strides |
| Basic slicing, such as `x[:, 1:]` | Share storage; may change the starting position and strides |
| `view(*shape)` | Share storage or raise an error if the layout is incompatible |
| `reshape(*shape)` | Share storage when possible; allow a copy when necessary |
| `expand(*shape)` | Share storage using zero strides along expanded dimensions |
| Arithmetic and matrix multiplication | Produce independent result storage; leave inputs unchanged |
| Reductions, such as `sum()` | Produce result storage; leave inputs unchanged |
| `Tensor.stack(tensors, axis=0)` | Produce independent storage and retain gradient connections when tracking is enabled |
| Advanced indexing with index arrays or boolean masks | Produce independent storage |
| `clone()` | Always create independent storage containing the same values |
| `contiguous()` | Return the tensor itself if already contiguous; otherwise copy into contiguous storage |
| `numpy()` | Return an independent NumPy copy with the same values, shape, and dtype; strides and storage layout may differ |

Only add these methods as they are implemented. The table defines their intended behavior.

`view` and `reshape` preserve the number of elements and the logical traversal order. They are not aliases. Shapes may contain one `-1` for inference; invalid sizes, multiple inferred dimensions, and ambiguous inference raise an error. A transpose validates the requested axes and preserves each element's identity: for a matrix, `x.transpose(0, 1)[j, i]` corresponds to `x[i, j]`.

For example, a contiguous `(2, 3)` matrix can be viewed as `(6,)`. Transposing that matrix gives shape `(3, 2)` with a different traversal order. Flattening the transpose in row order requires a copy, so `view(6)` must fail while `reshape(6)` may succeed by copying. Some other noncontiguous layouts still permit particular views; contiguity is sufficient, not always necessary.

Broadcasting aligns dimensions from the right; each aligned pair must match or include a dimension of size one. It reuses the smaller input without duplicating it. An expanded view can map several logical elements to one stored value. Arithmetic using that view still allocates its output.

## NumPy export and debugging the actual storage

`tensor.numpy()` exports independent values. The returned array must have the same values, shape, and dtype, but it does not have to preserve the tensor's strides, offset, contiguity, or storage-sharing relationships. Copying the values does not change the original tensor's layout.

Plain `array.copy()` defaults to C order, producing a packed row-major result. Using `array.copy(order="K")` preserves the layout as closely as possible, such as retaining a transposed matrix's column-major layout. It still does not guarantee identical strides: gaps in a stepped slice and repeated addresses in a broadcast view need not survive the copy. See [NumPy's copy order options](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.copy.html).

For example, this standalone NumPy code demonstrates why an exported copy cannot stand in for the original layout:

```python
import numpy as np

source = np.arange(12, dtype=np.float32)  # 12 elements, 48 element-storage bytes
view = source[::2]                      # [0, 2, 4, 6, 8, 10]
exported = view.copy(order="K")

print(view.strides)                      # (8,) bytes: gaps between selected values
print(exported.strides)                  # (4,) bytes: copied values packed together
print(view.flags.c_contiguous)           # False
print(exported.flags.c_contiguous)       # True
print(view.nbytes, exported.nbytes)      # 24 24: six logical float32 elements each
print(np.shares_memory(view, source))    # True: view retains the original allocation
print(np.shares_memory(exported, source)) # False: export has independent storage
```

Both arrays report 24 bytes through `nbytes`, but their storage relationships differ. The view keeps source's 48-byte allocation alive; the copy has its own 24-byte element buffer. These figures exclude Python objects, allocator overhead, and other process memory. NumPy's [`nbytes`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.nbytes.html) reports logical element bytes, not the size of the backing allocation a view retains.

Use `tensor.strides`, `tensor.shape`, and `tensor.is_contiguous()` to inspect the actual backing array. `tensor.numpy().strides` and the exported array's contiguity flags describe the copy instead. A future storage-inspection method could report offsets, logical bytes, and backing allocation details directly; it would not need to copy or expose writable numerical storage. Exporting values and inspecting storage are separate debugging needs.

## Shared storage, lifetime, and mutation

NumPy keeps backing storage alive while an array view references it. A view remains valid after the original Tensor object is deleted. A small slice may keep a much larger allocation alive; clone it when independent, smaller storage is needed.

The first public API uses operations that leave existing values unchanged. It does not expose a writable backing array or support element assignment and in-place arithmetic. `numpy()` copies specifically so editing the exported array cannot mutate a tensor through an external alias. There is no automatic copy when a shared value is written: if mutation is added later, every view observing that storage must see the change.

Internal code must treat operation inputs as unchanged. Expanded views can contain overlapping positions and may be read-only NumPy arrays; they must not be used as writable outputs.

Public mutation needs an explicit design before it is added, especially with autograd. Restricting mutation initially lets us use shared views without immediately implementing mutation tracking.

## Autograd contracts

Autograd means automatic differentiation: the framework records calculations so it can later calculate how a final value, usually a loss, changes when an input changes. The forward pass calculates values; the backward pass calculates their derivatives, called gradients. A computation graph records the tensors and operations connecting them.

These are the rules for the future autograd implementation. The examples describe intended behavior.

**1. Record dependencies when gradients are requested.**

In `y = x * 2`, if `x` requires gradients, `y` must remember that it came from multiplying `x` by two. Backward uses that relationship to send a gradient to `x`. A freshly constructed input is a **leaf**: it starts a graph and has no earlier operation history. A leaf can still require gradients.

The private array wrapper must let an operation attach the information needed for backward, such as its inputs and derivative rule. Reusing an array must not accidentally remove the operation's history.

Construction is deliberately separate from tracked operations. `Tensor([a, b])` stacks only values and creates a fresh leaf. `Tensor.stack([a, b])` records the inputs so a future backward implementation can reach them. They can contain the same numbers while having different gradient histories.

**2. Treat storage and gradient history as separate relationships.**

Storage determines where the numbers live. Gradient history determines which earlier tensors those numbers depend on. Sharing storage does not imply disconnecting history, and copying storage does not imply disconnecting history either.

- `clone()` creates independent storage with the same values. If the input requires gradients, the clone remains connected to it. Copying each value unchanged has derivative one, so the clone sends its incoming gradient back to the input.
- `detach()` creates a tensor that shares the input's storage but has no connection to its earlier gradient history and does not require gradients initially. Calculations through the detached tensor do not send gradients back to the original. Detach does not make a memory copy or protect shared values from mutation.
- `detach().clone()` gives independent storage without a gradient connection to the original.

| Operation | Shares input storage? | Keeps a gradient connection to the input when tracking is enabled? |
| --- | --- | --- |
| `clone()` | No | Yes |
| `detach()` | Yes | No |
| `transpose()` | Yes | Yes |
| `reshape()` | Sometimes | Yes, whether it shares storage or copies |
| `Tensor([a, b])` | No | No; construction creates a fresh leaf |
| `Tensor.stack([a, b])` | No | Yes |
| `detach().clone()` | No | No |

For example, if `x` contains `3` and requires gradients, `y = x.clone()` followed by `loss = (y * y).sum()` must give `x` a gradient of `6` after backward. Although `y` has separate storage, the loss still depends on `x`. Replacing clone with detach removes that connection; the resulting calculation alone has no tracked path back to `x`.

**3. Give each tensor a gradient with the same shape as its values.**

A tensor of shape `(2, 3)` has six input values. Its gradient also has shape `(2, 3)`, containing one derivative for each value. Gradients start absent until needed; once calculated, their shapes must match their tensors.

**4. Reverse shape changes correctly during backward.**

Backward through transpose applies the inverse axis permutation. Backward through reshape restores the input shape. Both rules route derivatives to the corresponding input positions, even when forward used shared storage.

Broadcasting reuses input values, so backward must add their contributions. If a bias of shape `(3,)` was used across two rows, the output gradient might be:

```text
[[1, 2, 3],
 [4, 5, 6]]
```

The bias gradient must sum across those rows to become `[5, 7, 9]`, with the original shape `(3,)`.

**5. Accumulate contributions from every path in the graph.**

In `loss = x * 2 + x * 3`, `x` affects the loss through two paths. Its gradient is `2 + 3 = 5`. Backward must add both contributions rather than overwrite one with the other. This rule concerns paths within a backward calculation; how gradients are cleared or accumulated across separate backward calls needs its own explicit policy.

**6. Use the values from the forward calculation when backward needs them.**

For `loss = x * x` calculated at `x = 3`, the gradient is `6`. If shared storage changes to `10` before backward, using that new value would incorrectly produce `20`.

The first API avoids public mutation. Before adding writes or shared external imports, we must decide how to preserve needed values or detect that they changed and raise an error. Mutation tracking detects a change; it does not itself recover the original values. Only changes to values actually needed by backward invalidate that calculation.

An optimizer updates model parameters using their gradients. Updates can occur after the backward calculations that need those parameters' old values have finished. Changing values through a detached tensor still changes shared storage, so detaching must not bypass any future checks protecting saved values.

## Alternatives

| Design | Benefit | Cost |
| --- | --- | --- |
| Copy input and every operation result | Simple ownership; results have independent storage | Copies views and duplicates newly allocated results |
| Reuse compatible arrays everywhere, including public input | Avoids input copies too | External code can change tensor values through the original array |
| Copy public input; wrap internal results — our choice | Predictable construction and efficient internal views | View behavior and mutation rules must be explicit |
| Offer explicit shared imports alongside copying construction | Callers can choose ownership for large external arrays | More API choices and external mutation concerns |

Using `np.asarray` everywhere would adopt the second design: it can reuse a compatible array, but can also allocate for lists or dtype conversions. It is not an unconditional zero-copy guarantee. The internal wrapping path should require an already suitable array instead.

An explicit shared import can be added later. Dtype conversion or transfer to a different device generally requires copying; an API promising zero copy must reject incompatible input rather than silently convert it.

## How PyTorch compares

PyTorch exposes both copying construction and external sharing:

| API | Behavior |
| --- | --- |
| `torch.tensor(data)` | Copies input and creates a leaf without prior autograd history |
| `torch.from_numpy(array)` | Shares a supported NumPy array's CPU storage |
| `torch.as_tensor(data)` | Reuses data and preserves history where possible; conversion can require a copy |

Our constructor follows the ownership policy of [`torch.tensor`](https://docs.pytorch.org/docs/stable/generated/torch.tensor.html). PyTorch additionally provides [`from_numpy`](https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html) and [`as_tensor`](https://docs.pytorch.org/docs/stable/generated/torch.as_tensor.html) for callers who want sharing.

PyTorch's transpose and basic slicing produce views. Its reshape may copy, and contiguous copies only when needed. Our intended contracts follow these [tensor view semantics](https://docs.pytorch.org/docs/stable/tensor_view.html). Its [`view`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html) also requires a shape compatible with the existing strides.

One deliberate difference is export: PyTorch's [`numpy()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numpy.html) shares storage when its documented conditions are met. Our initial `numpy()` returns a copy.

PyTorch supports in-place operations and checks saved tensor version counters to detect changes that would invalidate backward. We can start with operations that leave inputs unchanged and add mutation support later. Those checks are part of its [autograd design](https://docs.pytorch.org/docs/stable/notes/autograd.html), not something that NumPy storage handles for us.

## Verification when implementing

- Changing constructor input leaves the tensor unchanged.
- Internal wrapping retains the exact supplied array and does not cast or copy it.
- Transpose and nonempty basic slices share storage and produce the expected values and strides.
- A compatible view shares storage; an incompatible view fails; reshape preserves values even when copying is necessary.
- Arithmetic, clone, and NumPy export have independent storage and leave inputs unchanged.
- NumPy export preserves values, shape, and dtype without promising identical strides or contiguity. Storage inspection reads the tensor's metadata rather than the export's layout.
- Contiguous returns itself when possible and produces a contiguous copy when needed.
- Views remain valid after the original Tensor object is deleted.
- Scalars, empty arrays, invalid shapes, and invalid axes obey their contracts. Empty arrays require checking backing references and layout because `np.shares_memory` cannot demonstrate overlap when there are no elements.
- When autograd is added, verify shape changes and broadcast reductions with numerical gradient checks.
