import torch
import pytest

from torchmil.data.representations import seq_to_spatial, spatial_to_seq

def test_seq_to_spatial_basic():
    # Test case 1: Basic test with small input tensors
    X = torch.tensor([[
        [1, 2],
        [3, 4],
        [5, 6]
    ]], dtype=torch.float32)
    coords = torch.tensor([[
        [0, 0],
        [1, 0],
        [0, 1]
    ]], dtype=torch.long)
    X_esp = seq_to_spatial(X, coords)
    expected_shape = (1, 2, 2, 2)  # (batch_size, coord1_max+1, coord2_max+1, dim)
    assert X_esp.shape == expected_shape
    expected_result = torch.tensor([[
        [
            [1, 2],
            [5, 6]
        ],
        [
            [3, 4],
            [0, 0]
        ]
    ]], dtype=torch.float32)
    assert torch.allclose(X_esp, expected_result)

def test_seq_to_spatial_multiple_batches():
    # Test case 2: Multiple batches
    X = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32)
    coords = torch.tensor([[[0, 0], [1, 0]], [[0, 1], [1, 1]]], dtype=torch.long)
    X_esp = seq_to_spatial(X, coords)
    expected_shape = (2, 2, 2, 2)
    assert X_esp.shape == expected_shape
    expected_result = torch.tensor(
        [[[[1., 2.],
          [0., 0.]],

         [[3., 4.],
          [0., 0.]]],


        [[[0., 0.],
          [5., 6.]],

         [[0., 0.],
          [7., 8.]]]]
        , dtype=torch.float32)
    assert torch.allclose(X_esp, expected_result)

def test_seq_to_spatial_different_dim():
    # Test case 3: Different dimension of X
    X = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32)
    coords = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)
    X_esp = seq_to_spatial(X, coords)
    expected_shape = (1, 2, 2, 3)
    assert X_esp.shape == expected_shape
    expected_result = torch.tensor([[[[0., 0., 0.], [1., 2., 3.]],
                                     [[4., 5., 6.], [0., 0., 0.]]]], dtype=torch.float32)
    assert torch.allclose(X_esp, expected_result)

def test_seq_to_spatial_large_coordinates():
    # Test case 4: Large coordinate values
    X = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)
    coords = torch.tensor([[[10, 5], [20, 15]]], dtype=torch.long)
    X_esp = seq_to_spatial(X, coords)
    expected_shape = (1, 21, 16, 2)
    assert X_esp.shape == expected_shape
    expected_result = torch.zeros(expected_shape, dtype=torch.float32)
    expected_result[0, 10, 5, :] = X[0, 0, :]
    expected_result[0, 20, 15, :] = X[0, 1, :]
    assert torch.allclose(X_esp, expected_result)

def test_seq_to_spatial_3d_coordinates():
    # Test case 5: 3D coordinates
    X = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)
    coords = torch.tensor([[[0, 0, 0], [1, 0, 1]]], dtype=torch.long)
    X_esp = seq_to_spatial(X, coords)
    expected_shape = (1, 2, 1, 2, 2)
    assert X_esp.shape == expected_shape
    expected_result = torch.zeros(expected_shape, dtype=torch.float32)
    expected_result[0, 0, 0, 0, :] = X[0, 0, :]
    expected_result[0, 1, 0, 1, :] = X[0, 1, :]
    assert torch.allclose(X_esp, expected_result)

def test_seq_to_spatial_bag_size_one():
    # Test case 7: bag_size is 1
    X = torch.tensor([[[1, 2]]], dtype=torch.float32)
    coords = torch.tensor([[[0, 0]]], dtype=torch.long)
    X_esp = seq_to_spatial(X, coords)
    expected_shape = (1, 1, 1, 2)
    assert X_esp.shape == expected_shape
    expected_result = torch.tensor([[[[1, 2]]]], dtype=torch.float32)
    assert torch.allclose(X_esp, expected_result)

def test_seq_to_spatial_coord_max_zero():
    # Test case 8: max coordinate is 0
    X = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)
    coords = torch.tensor([[[0, 0], [0, 0]]], dtype=torch.long)
    X_esp = seq_to_spatial(X, coords)
    expected_shape = (1, 1, 1, 2)
    assert X_esp.shape == expected_shape
    expected_result = torch.tensor([[[[3, 4]]]], dtype=torch.float32)
    assert torch.allclose(X_esp, expected_result)

def test_spatial_to_seq_basic():
    # Test case 1: Basic test with small input tensors
    X_esp = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32)
    coords = torch.tensor([[[0, 0], [1, 0]]], dtype=torch.long)
    X_seq = spatial_to_seq(X_esp, coords)
    expected_shape = (1, 2, 2)
    assert X_seq.shape == expected_shape
    expected_result = torch.tensor([[[1, 2], [5, 6]]], dtype=torch.float32)
    assert torch.allclose(X_seq, expected_result)

def test_spatial_to_seq_multiple_batches():
    # Test case 2: Multiple batches
    X_esp = torch.tensor(
        [[[[1., 2.], [3., 4.]],
          [[0., 0.], [0., 0.]]],
         [[[0., 0.], [5., 6.]],
          [[0., 0.], [7., 8.]]]]
        , dtype=torch.float32)
    coords = torch.tensor([[[0, 0], [1, 0]], [[0, 1], [1, 1]]], dtype=torch.long)
    X_seq = spatial_to_seq(X_esp, coords)
    expected_shape = (2, 2, 2)
    assert X_seq.shape == expected_shape
    expected_result = torch.tensor([[[1, 2], [0, 0]], [[5, 6], [7, 8]]], dtype=torch.float32)
    assert torch.allclose(X_seq, expected_result)

def test_spatial_to_seq_different_dim():
    # Test case 3: Different dimension of X
    X_esp = torch.tensor([[[[0., 0., 0.], [1., 2., 3.]],
                             [[4., 5., 6.], [0., 0., 0.]]]], dtype=torch.float32)
    coords = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)
    X_seq = spatial_to_seq(X_esp, coords)
    expected_shape = (1, 2, 3)
    assert X_seq.shape == expected_shape
    expected_result = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32)
    assert torch.allclose(X_seq, expected_result)

def test_spatial_to_seq_large_coordinates():
    # Test case 4: Large coordinate values
    X_esp = torch.zeros((1, 21, 16, 2), dtype=torch.float32)
    X_esp[0, 10, 5, :] = torch.tensor([1, 2], dtype=torch.float32)
    X_esp[0, 20, 15, :] = torch.tensor([3, 4], dtype=torch.float32)
    coords = torch.tensor([[[10, 5], [20, 15]]], dtype=torch.long)
    X_seq = spatial_to_seq(X_esp, coords)
    expected_shape = (1, 2, 2)
    assert X_seq.shape == expected_shape
    expected_result = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)
    assert torch.allclose(X_seq, expected_result)

def test_spatial_to_seq_3d_coordinates():
    # Test case 5: 3D coordinates
    X_esp = torch.zeros((1, 2, 1, 2, 2), dtype=torch.float32)
    X_esp[0, 0, 0, 0, :] = torch.tensor([1, 2], dtype=torch.float32)
    X_esp[0, 1, 0, 1, :] = torch.tensor([3, 4], dtype=torch.float32)

    coords = torch.tensor([[[0, 0, 0], [1, 0, 1]]], dtype=torch.long)
    X_seq = spatial_to_seq(X_esp, coords)
    expected_shape = (1, 2, 2)
    assert X_seq.shape == expected_shape
    expected_result = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)
    assert torch.allclose(X_seq, expected_result)

def test_spatial_to_seq_empty_input():
    # Test case 6: Empty input tensor
    X_esp = torch.empty(0, 0, 0, 2, dtype=torch.float32)
    coords = torch.empty(0, 0, 2, dtype=torch.long)
    X_seq = spatial_to_seq(X_esp, coords)
    expected_shape = (0, 0, 2)
    assert X_seq.shape == expected_shape
    assert torch.allclose(X_seq, torch.empty(expected_shape, dtype=torch.float32))

def test_spatial_to_seq_bag_size_one():
    # Test case 7: bag_size is 1
    X_esp = torch.tensor([[[[1, 2]]]], dtype=torch.float32)
    coords = torch.tensor([[[0, 0]]], dtype=torch.long)
    X_seq = spatial_to_seq(X_esp, coords)
    expected_shape = (1, 1, 2)
    assert X_seq.shape == expected_shape
    expected_result = torch.tensor([[[1, 2]]], dtype=torch.float32)
    assert torch.allclose(X_seq, expected_result)

def test_spatial_to_seq_coord_max_zero():
    # Test case 8: Test when the maximum coordinate value is 0.
    X_esp = torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.float32)
    coords = torch.tensor([[[0, 0], [0, 0]]], dtype=torch.long)
    X_seq = spatial_to_seq(X_esp, coords)
    expected_shape = (1, 2, 2)
    assert X_seq.shape == expected_shape
    expected_result = torch.tensor([[[1, 2], [1, 2]]], dtype=torch.float32)
    assert torch.allclose(X_seq, expected_result)
