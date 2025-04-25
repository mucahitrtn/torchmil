import torch
import pytest
from torchmil.nn import MeanPool

@pytest.fixture
def meanpool_instance():
    return MeanPool()

def test_meanpool_no_mask(meanpool_instance):
    batch_size = 2
    bag_size = 3
    in_dim = 4
    X = torch.randn(batch_size, bag_size, in_dim)
    z = meanpool_instance(X)
    assert z.shape == (batch_size, in_dim)
    for i in range(batch_size):
        for j in range(in_dim):
            assert torch.allclose(z[i, j], torch.mean(X[i, :, j]))

def test_meanpool_with_mask(meanpool_instance):
    batch_size = 2
    bag_size = 3
    in_dim = 4
    X = torch.randn(batch_size, bag_size, in_dim)
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    z = meanpool_instance(X, mask)
    assert z.shape == (batch_size, in_dim)
    for i in range(batch_size):
        for j in range(in_dim):
            masked_values = X[i, mask[i], j]
            if len(masked_values) > 0:
                assert torch.allclose(z[i, j], torch.mean(masked_values))
            else:
                assert torch.allclose(z[i,j], torch.tensor(0.0))

def test_meanpool_empty_mask(meanpool_instance):
    batch_size = 2
    bag_size = 3
    in_dim = 4
    X = torch.randn(batch_size, bag_size, in_dim)
    mask = torch.zeros(batch_size, bag_size, dtype=torch.bool)
    z = meanpool_instance(X, mask)
    assert z.shape == (batch_size, in_dim)
    assert torch.allclose(z, torch.zeros(batch_size, in_dim))

def test_meanpool_all_mask(meanpool_instance):
    batch_size = 2
    bag_size = 3
    in_dim = 4
    X = torch.randn(batch_size, bag_size, in_dim)
    mask = torch.ones(batch_size, bag_size, dtype=torch.bool)
    z = meanpool_instance(X, mask)
    assert z.shape == (batch_size, in_dim)
    for i in range(batch_size):
        for j in range(in_dim):
            assert torch.allclose(z[i, j], torch.mean(X[i, :, j]))
