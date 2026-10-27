"""Public Tensor contracts; private storage is inspected only for ownership checks."""

import operator

import numpy as np
import pytest

from tinytorch import Tensor


class TestConstruction:
    @pytest.mark.parametrize(
        "data",
        [
            3,
            3.5,
            "3.5",
            ["1", "2"],
            [1, "2"],
            b"3",
            True,
            np.array([True, False]),
            np.array([1, 2], dtype=object),
            np.int64(3),
            np.float32(3),
            [1, 2, 3],
            ((1, 2), (3, 4)),
            [],
            [[], []],
            np.array([1, 2], dtype=np.uint32),
            np.array([1.5, 2.5], dtype=np.float64),
            np.asfortranarray([[1, 2], [3, 4]], dtype=np.float32),
        ],
    )
    def test_constructor_accepts_real_numeric_input_without_metadata_methods(self, data):
        t = Tensor(data)
        assert t._storage.dtype == np.dtype(np.float32)
        assert t._storage.flags.c_contiguous
        np.testing.assert_array_equal(t._storage, np.asarray(data, dtype=np.float32))
        if isinstance(data, np.ndarray):
            assert not np.shares_memory(t._storage, data)

    def test_constructor_copies_tensor_input_without_inheriting_gradient_request(self):
        original = Tensor([[1, 2], [3, 4]], requires_grad=True)
        copied = Tensor(original)
        assert copied._storage is not original._storage
        assert not np.shares_memory(copied._storage, original._storage)
        assert copied._storage.flags.c_contiguous
        np.testing.assert_array_equal(copied._storage, original._storage)
        assert copied._requires_grad is False
        assert original._requires_grad is True

    @pytest.mark.parametrize("container", [list, tuple])
    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_constructor_stacks_tensor_values_as_a_fresh_leaf(self, container, requires_grad):
        a = Tensor([1, 2], requires_grad=True)
        b = Tensor([3, 4], requires_grad=True)
        result = Tensor(container([a, b]), requires_grad=requires_grad)
        np.testing.assert_array_equal(result._storage, [[1, 2], [3, 4]])
        assert result._storage.dtype == np.dtype(np.float32)
        assert result._storage.flags.c_contiguous
        assert not np.shares_memory(result._storage, a._storage)
        assert not np.shares_memory(result._storage, b._storage)
        assert result._requires_grad is requires_grad
        assert a._requires_grad is True and b._requires_grad is True
        # No earlier operation context is inherited; autograd is not implemented yet.
        assert getattr(result, "_grad_fn", None) is None
        assert getattr(result, "_ctx", None) is None
        a._storage[0] = 99
        np.testing.assert_array_equal(result._storage, [[1, 2], [3, 4]])

    def test_constructor_stacking_packs_transposed_inputs(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        b = Tensor([[7, 8, 9], [10, 11, 12]])
        a = a.transpose(0, 1)
        b = b.transpose(0, 1)
        result = Tensor([a, b])
        assert result._storage.flags.c_contiguous
        assert result._storage.strides == (24, 8, 4)
        np.testing.assert_array_equal(result._storage, [[[1, 4], [2, 5], [3, 6]], [[7, 10], [8, 11], [9, 12]]])
        assert not np.shares_memory(result._storage, a._storage)
        assert not np.shares_memory(result._storage, b._storage)

    def test_constructor_stacks_scalar_tensors(self):
        result = Tensor([Tensor(1), Tensor(2)])
        assert result._storage.shape == (2,)
        np.testing.assert_array_equal(result._storage, [1, 2])

    def test_constructor_stacks_empty_tensors(self):
        result = Tensor([Tensor([]), Tensor([])])
        assert result._storage.shape == (2, 0)
        assert result._storage.flags.c_contiguous

    def test_constructor_stacking_requires_equal_input_shapes(self):
        with pytest.raises(ValueError):
            Tensor([Tensor([1, 2]), Tensor([3])])

    @pytest.mark.parametrize("data_kind", ["tensor_first", "number_first"])
    def test_constructor_rejects_mixed_tensor_and_numeric_lists(self, data_kind):
        t = Tensor([1, 2])
        data = [t, 3] if data_kind == "tensor_first" else [3, t]
        with pytest.raises((TypeError, ValueError)):
            Tensor(data)

    @pytest.mark.parametrize("kwargs", [{"device": "cuda"}, {"dtype": "float64"}])
    def test_constructor_rejects_unsupported_device_and_dtype(self, kwargs):
        with pytest.raises(NotImplementedError):
            Tensor([1, 2, 3], **kwargs)

    def test_metadata_describes_the_array(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        assert t.shape == (2, 3)
        assert t.ndim == 2
        assert t.numel() == 6
        assert t.dtype == "float32"
        assert t.device == "cpu"
        assert t.strides == (12, 4)
        assert t.is_contiguous()
        np.testing.assert_array_equal(t.numpy(), [[1, 2, 3], [4, 5, 6]])

    def test_construction_copies_even_a_float32_array(self):
        source = np.array([1, 2, 3], dtype=np.float32)
        t = Tensor(source)
        assert not np.shares_memory(source, t._storage)
        source[0] = 99
        np.testing.assert_array_equal(t.numpy(), [1, 2, 3])

    def test_construction_packs_column_major_input_in_row_order(self):
        source = np.asfortranarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        t = Tensor(source)
        assert t.is_contiguous()
        assert t.strides == (12, 4)
        assert not np.shares_memory(source, t._storage)
        np.testing.assert_array_equal(t.numpy(), source)

    @pytest.mark.parametrize("data", [3.0, [3.0], [], [[], []]])
    def test_scalar_vector_and_empty_shapes(self, data):
        expected = np.array(data, dtype=np.float32)
        t = Tensor(data)
        assert t.shape == expected.shape
        assert t.ndim == expected.ndim
        assert t.numel() == expected.size

    def test_tuple_input(self):
        t = Tensor(((1, 2), (3, 4)))
        np.testing.assert_array_equal(t.numpy(), [[1, 2], [3, 4]])

    @pytest.mark.parametrize(
        "data",
        [
            [[1, 2], [3]],
            ["hello", "world"],
            1 + 2j,
            None,
            {1, 2},
        ],
    )
    def test_invalid_input_is_rejected(self, data):
        with pytest.raises((TypeError, ValueError)):
            Tensor(data)

    @pytest.mark.parametrize(
        "array",
        [
            np.array(3, dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32).T,
            np.broadcast_to(np.array([10, 20, 30], dtype=np.float32), (2, 3)),
        ],
    )
    def test_internal_wrapper_retains_array_and_initializes_fields(self, array):
        t = Tensor._from_storage(array)
        assert t._storage is array
        assert t._device == "cpu"
        assert t._requires_grad is False

    def test_internal_wrapper_bypasses_public_construction(self):
        class ConstructorMustNotRun(Tensor):
            def __init__(self, *args, **kwargs):
                raise AssertionError("Internal wrapping must not call __init__")

        array = np.array([1, 2, 3], dtype=np.float32)
        t = ConstructorMustNotRun._from_storage(array)
        assert isinstance(t, ConstructorMustNotRun)
        assert t._storage is array

    def test_internal_wrapper_preserves_the_exact_transposed_array(self):
        source = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        array = source.T
        t = Tensor._from_storage(array)
        assert t._storage is array
        assert t.shape == (3, 2)
        assert t.strides == (4, 12)
        assert not t.is_contiguous()
        assert np.shares_memory(t._storage, source)

    def test_internal_wrapper_preserves_read_only_broadcast_views(self):
        source = np.array([10, 20, 30], dtype=np.float32)
        array = np.broadcast_to(source, (2, 3))
        t = Tensor._from_storage(array)
        assert t._storage is array
        assert t.strides == (0, 4)
        assert not t._storage.flags.writeable

    @pytest.mark.parametrize("array, error", [
        (np.array([1], dtype=np.float64), NotImplementedError),
        (np.array([1], dtype=np.int32), NotImplementedError),
        ([1.0], TypeError),
    ])
    def test_internal_wrapper_rejects_conversion_instead_of_copying(self, array, error):
        with pytest.raises(error):
            Tensor._from_storage(array)

    def test_numpy_export_cannot_modify_tensor_storage(self):
        t = Tensor([1, 2, 3])
        exported = t.numpy()
        assert not np.shares_memory(exported, t._storage)
        exported[0] = 99
        np.testing.assert_array_equal(t.numpy(), [1, 2, 3])


class TestArithmetic:
    @pytest.mark.parametrize("left_shape, right_shape", [
        ((2, 1, 3), (1, 4, 1)), ((), (2, 3)), ((0, 3), (3,)),
    ])
    def test_broadcasting_across_ranks_and_empty_dimensions(self, left_shape, right_shape):
        left = np.arange(np.prod(left_shape, dtype=int), dtype=np.float32).reshape(left_shape)
        right = np.ones(right_shape, dtype=np.float32)
        result = Tensor(left) + Tensor(right)
        np.testing.assert_array_equal(result.numpy(), left + right)
        assert result.dtype == "float32"

    def test_batched_matrix_multiplication(self):
        left = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        right = np.arange(20, dtype=np.float32).reshape(4, 5)
        result = Tensor(left) @ Tensor(right)
        assert result.shape == (2, 3, 5)
        np.testing.assert_allclose(result.numpy(), left @ right)

    @pytest.mark.parametrize(
        "op, expected",
        [
            (operator.add, [12, 24]),
            (operator.sub, [-8, -16]),
            (operator.mul, [20, 80]),
            (operator.truediv, [0.2, 0.2]),
        ],
    )
    def test_tensor_operations_create_independent_results(self, op, expected):
        a, b = Tensor([2, 4]), Tensor([10, 20])
        result = op(a, b)
        np.testing.assert_allclose(result.numpy(), expected)
        assert result.dtype == "float32"
        assert not np.shares_memory(result._storage, a._storage)
        assert not np.shares_memory(result._storage, b._storage)
        np.testing.assert_array_equal(a.numpy(), [2, 4])
        np.testing.assert_array_equal(b.numpy(), [10, 20])

    @pytest.mark.parametrize(
        "op, left, expected",
        [
            (operator.add, False, [4, 6]),
            (operator.add, True, [4, 6]),
            (operator.sub, False, [0, 2]),
            (operator.sub, True, [0, -2]),
            (operator.mul, False, [4, 8]),
            (operator.mul, True, [4, 8]),
            (operator.truediv, False, [1, 2]),
            (operator.truediv, True, [1, 0.5]),
        ],
    )
    def test_scalar_operands_on_either_side(self, op, left, expected):
        t = Tensor([2, 4])
        result = op(2, t) if left else op(t, 2)
        np.testing.assert_allclose(result.numpy(), expected)
        assert result.dtype == "float32"

    def test_bias_broadcast_addition(self):
        x, bias = Tensor([[1, 2, 3], [4, 5, 6]]), Tensor([10, 20, 30])
        result = x + bias
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result.numpy(), [[11, 22, 33], [14, 25, 36]])
        np.testing.assert_array_equal(bias.numpy(), [10, 20, 30])

    def test_incompatible_broadcast_is_rejected(self):
        x, y = Tensor([[1, 2, 3], [4, 5, 6]]), Tensor([10, 20])
        with pytest.raises((TypeError, ValueError)):
            x + y

    def test_matrix_multiplication(self):
        a, b = Tensor([[1, 2]]), Tensor([[3], [4]])
        result = a @ b
        assert result.shape == (1, 1)
        np.testing.assert_array_equal(result.numpy(), [[11]])


class TestViews:
    def test_negative_stride_slice_and_export(self):
        x = Tensor([1, 2, 3])
        reversed_x = x[::-1]
        assert reversed_x.strides == (-4,)
        assert np.shares_memory(reversed_x._storage, x._storage)
        np.testing.assert_array_equal(reversed_x.numpy(), [3, 2, 1])
        assert not np.shares_memory(reversed_x.numpy(), x._storage)

    def test_clone_preserves_broadcast_values_without_sharing(self):
        source = Tensor([1, 2]).expand(3, 2)
        copied = source.clone()
        assert not np.shares_memory(copied._storage, source._storage)
        np.testing.assert_array_equal(copied.numpy(), [[1, 2], [1, 2], [1, 2]])

    def test_transpose_preserves_storage_and_swaps_strides(self):
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        t = x.transpose(0, 1)
        assert t.shape == (3, 2)
        assert t.strides == (4, 12)
        assert not t.is_contiguous()
        assert np.shares_memory(t._storage, x._storage)
        np.testing.assert_array_equal(t.numpy(), [[1, 4], [2, 5], [3, 6]])

    def test_negative_transpose_axes(self):
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(x.transpose(-1, -2).numpy(), [[1, 4], [2, 5], [3, 6]])

    def test_permute_reorders_all_axes(self):
        x = Tensor(np.arange(24).reshape(2, 3, 4))
        p = x.permute(2, 0, 1)
        assert p.shape == (4, 2, 3)
        assert p.strides == (4, 48, 16)
        assert np.shares_memory(p._storage, x._storage)
        np.testing.assert_array_equal(p.numpy(), x.numpy().transpose(2, 0, 1))
        np.testing.assert_array_equal(x.permute((2, 0, 1)).numpy(), p.numpy())

    def test_invalid_axes_are_rejected(self):
        x = Tensor([[1, 2], [3, 4]])
        with pytest.raises((IndexError, ValueError)):
            x.transpose(0, 2)
        with pytest.raises((IndexError, ValueError)):
            x.permute(0, 0)

    def test_basic_slice_keeps_original_row_stride_and_advances_offset(self):
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = x[1:, :2]
        assert y.shape == (1, 2)
        assert y.strides == (12, 4)
        assert y._storage.ctypes.data - x._storage.ctypes.data == 12
        assert np.shares_memory(y._storage, x._storage)
        assert y.is_contiguous()  # singleton row axis: its stride is unused
        np.testing.assert_array_equal(y.numpy(), [[4, 5]])

    def test_two_row_slice_is_not_contiguous(self):
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = x[:, :2]
        assert y.strides == (12, 4)
        assert not y.is_contiguous()
        assert np.shares_memory(y._storage, x._storage)
        np.testing.assert_array_equal(y.numpy(), [[1, 2], [4, 5]])

    def test_advanced_indexing_copies(self):
        x = Tensor([10, 20, 30, 40])
        y = x[[2, 0, 2]]
        assert not np.shares_memory(y._storage, x._storage)
        np.testing.assert_array_equal(y.numpy(), [30, 10, 30])

    def test_element_selection_returns_a_scalar_tensor(self):
        x = Tensor([[1, 2], [3, 4]])
        y = x[1, 0]
        assert isinstance(y, Tensor)
        assert y.shape == ()
        assert y.numel() == 1
        assert y.numpy().item() == 3

    def test_reshape_shares_a_compatible_layout(self):
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = x.reshape(3, 2)
        assert np.shares_memory(y._storage, x._storage)
        np.testing.assert_array_equal(y.numpy(), [[1, 2], [3, 4], [5, 6]])

    def test_reshape_copies_when_transpose_cannot_be_flattened_as_a_view(self):
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = x.transpose(0, 1).reshape(6)
        assert not np.shares_memory(y._storage, x._storage)
        np.testing.assert_array_equal(y.numpy(), [1, 4, 2, 5, 3, 6])

    def test_view_shares_storage_and_infers_one_dimension(self):
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = x.view(3, -1)
        assert y.shape == (3, 2)
        assert np.shares_memory(y._storage, x._storage)
        np.testing.assert_array_equal(x.reshape((3, 2)).numpy(), y.numpy())
        np.testing.assert_array_equal(x.view((3, 2)).numpy(), y.numpy())

    def test_view_rejects_a_layout_that_requires_copying(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]]).transpose(0, 1)
        with pytest.raises(ValueError):
            t.view(6)

    def test_view_can_share_a_noncontiguous_compatible_layout(self):
        source = Tensor(np.arange(12))
        stepped = source[::2]
        y = stepped.view(2, 3)
        assert not y.is_contiguous()
        assert y.strides == (24, 8)
        assert np.shares_memory(y._storage, source._storage)
        np.testing.assert_array_equal(y.numpy(), [[0, 2, 4], [6, 8, 10]])

    @pytest.mark.parametrize("shape", [(4, 2), (-1, -1), (-2, 3)])
    def test_invalid_reshape_shapes(self, shape):
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            x.reshape(*shape)

    def test_empty_view_is_valid_without_overlapping_element_bytes(self):
        x = Tensor([])
        y = x.view(0, 2)
        assert y.shape == (0, 2)
        assert y.numel() == 0
        assert not y._storage.flags.owndata

    def test_expand_reuses_bias_values_with_zero_stride(self):
        bias = Tensor([10, 20, 30])
        expanded = bias.expand(2, 3)
        assert expanded.shape == (2, 3)
        assert expanded.strides == (0, 4)
        assert expanded.numel() == 6
        assert np.shares_memory(expanded._storage, bias._storage)
        assert not expanded._storage.flags.writeable
        np.testing.assert_array_equal(expanded.numpy(), [[10, 20, 30], [10, 20, 30]])
        np.testing.assert_array_equal(bias.expand((2, 3)).numpy(), expanded.numpy())

    def test_expand_rejects_incompatible_shapes(self):
        bias = Tensor([10, 20, 30])
        with pytest.raises(ValueError):
            bias.expand(2, 2)
        with pytest.raises(ValueError):
            bias.expand(-1, 3)

    def test_clone_has_independent_storage(self):
        x = Tensor([[1, 2, 3], [4, 5, 6]]).transpose(0, 1)
        y = x.clone()
        assert y is not x
        assert not np.shares_memory(y._storage, x._storage)
        np.testing.assert_array_equal(y.numpy(), x.numpy())

    def test_contiguous_returns_self_or_a_packed_copy(self):
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        assert x.contiguous() is x
        t = x.transpose(0, 1)
        packed = t.contiguous()
        assert packed.is_contiguous()
        assert packed.strides == (8, 4)
        assert not np.shares_memory(packed._storage, x._storage)
        np.testing.assert_array_equal(packed.numpy(), t.numpy())

    def test_view_remains_valid_after_original_tensor_is_deleted(self):
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = x[:, :2]
        del x
        np.testing.assert_array_equal(y.numpy(), [[1, 2], [4, 5]])

    def test_public_element_assignment_is_not_supported(self):
        x = Tensor([1, 2, 3])
        with pytest.raises(TypeError):
            x[0] = 99
        np.testing.assert_array_equal(x.numpy(), [1, 2, 3])


class TestReductions:
    @pytest.mark.parametrize("method", ["sum", "mean", "max"])
    @pytest.mark.parametrize("axis", [-1, (0, 2), ()])
    def test_multiple_negative_and_empty_axis_tuples(self, method, axis):
        source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        output = getattr(Tensor(source), method)(axis=axis, keepdims=True)
        expected = getattr(np, method)(source, axis=axis, keepdims=True)
        np.testing.assert_allclose(output.numpy(), expected)
        assert output.shape == expected.shape

    def test_empty_sum_and_max(self):
        x = Tensor([])
        assert x.sum().numpy().item() == 0
        with pytest.raises(ValueError):
            x.max()

    @pytest.mark.parametrize(
        "method, axis, keepdims, expected",
        [
            ("sum", None, False, 21),
            ("sum", 0, False, [5, 7, 9]),
            ("sum", 1, False, [6, 15]),
            ("sum", 0, True, [[5, 7, 9]]),
            ("sum", 1, True, [[6], [15]]),
            ("mean", 1, False, [2, 5]),
            ("mean", 1, True, [[2], [5]]),
            ("max", 0, False, [4, 5, 6]),
            ("max", None, False, 6),
        ],
    )
    def test_values_shapes_and_storage(self, method, axis, keepdims, expected):
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = getattr(x, method)(axis=axis, keepdims=keepdims)
        expected_array = np.array(expected, dtype=np.float32)
        assert y.shape == expected_array.shape
        assert y.dtype == "float32"
        assert not np.shares_memory(y._storage, x._storage)
        np.testing.assert_allclose(y.numpy(), expected_array)
        np.testing.assert_array_equal(x.numpy(), [[1, 2, 3], [4, 5, 6]])

    def test_keepdims_allows_row_centering(self):
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        means = x.mean(axis=1, keepdims=True)
        result = x - means
        np.testing.assert_array_equal(result.numpy(), [[-1, 0, 1], [-1, 0, 1]])
