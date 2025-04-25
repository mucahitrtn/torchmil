import torch
import pytest
from torchmil.nn import MaxPool

@pytest.fixture
def maxpool_instance():
    return MaxPool()

def test_maxpool_no_mask(maxpool_instance):
    batch_size = 2
    bag_size = 3
    in_dim = 4
    X = torch.randn(batch_size, bag_size, in_dim)
    z = maxpool_instance(X)
    assert z.shape == (batch_size, in_dim)
    for i in range(batch_size):
        for j in range(in_dim):
            assert torch.allclose(z[i, j], torch.max(X[i, :, j]))

def test_maxpool_with_mask(maxpool_instance):
    batch_size = 2
    bag_size = 3
    in_dim = 4
    X = torch.randn(batch_size, bag_size, in_dim)
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    z = maxpool_instance(X, mask)
    assert z.shape == (batch_size, in_dim)

    for i in range(batch_size):
        for j in range(in_dim):
            masked_values = X[i, mask[i], j]
            if len(masked_values) > 0:
                assert torch.allclose(z[i, j], torch.max(masked_values))
            else:
                assert torch.allclose(z[i,j], torch.tensor(float('-inf')))

def test_maxpool_empty_mask(maxpool_instance):
    batch_size = 2
    bag_size = 3
    in_dim = 4
    X = torch.randn(batch_size, bag_size, in_dim)
    mask = torch.zeros(batch_size, bag_size, dtype=torch.bool)
    z = maxpool_instance(X, mask)
    assert z.shape == (batch_size, in_dim)
    assert torch.allclose(z, torch.full((batch_size, in_dim), float('-inf')))

def test_maxpool_all_mask(maxpool_instance):
    batch_size = 2
    bag_size = 3
    in_dim = 4
    X = torch.randn(batch_size, bag_size, in_dim)
    mask = torch.ones(batch_size, bag_size, dtype=torch.bool)
    z = maxpool_instance(X, mask)
    assert z.shape == (batch_size, in_dim)
    for i in range(batch_size):
        for j in range(in_dim):
            assert torch.allclose(z[i, j], torch.max(X[i, :, j]))
