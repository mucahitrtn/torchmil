import torch
import pytest
from torchmil.nn import Sm, ApproxSm, ExactSm

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def bag_size():
    return 3

@pytest.fixture
def in_dim():
    return 4

@pytest.fixture
def features(batch_size, bag_size, in_dim):
    return torch.randn(batch_size, bag_size, in_dim)

@pytest.fixture
def adjacency_matrix(batch_size, bag_size):
    return torch.randn(batch_size, bag_size, bag_size)

def test_approxsm_fixed_alpha(features, adjacency_matrix):
    approx_sm = ApproxSm(alpha=0.5, num_steps=5)
    result = approx_sm(features, adjacency_matrix)
    assert result.shape == features.shape

def test_approxsm_trainable_alpha(features, adjacency_matrix):
    approx_sm = ApproxSm(alpha='trainable', num_steps=5)
    result = approx_sm(features, adjacency_matrix)
    assert result.shape == features.shape
    assert approx_sm.coef.requires_grad

def test_exactsm_fixed_alpha(features, adjacency_matrix):
    exact_sm = ExactSm(alpha=0.5)
    result = exact_sm(features, adjacency_matrix)
    assert result.shape == features.shape

def test_exactsm_trainable_alpha(features, adjacency_matrix):
    exact_sm = ExactSm(alpha='trainable')
    result = exact_sm(features, adjacency_matrix)
    assert result.shape == features.shape
    assert exact_sm.coef.requires_grad

def test_sm_approx_mode(features, adjacency_matrix):
    sm = Sm(mode='approx', alpha=0.5, num_steps=5)
    result = sm(features, adjacency_matrix)
    assert result.shape == features.shape

def test_sm_exact_mode(features, adjacency_matrix):
    sm = Sm(mode='exact', alpha=0.5)
    result = sm(features, adjacency_matrix)
    assert result.shape == features.shape

def test_sm_invalid_mode():
    with pytest.raises(ValueError):
        Sm(mode='invalid')

def test_approxsm_single_feature_dim(batch_size, bag_size):
    features = torch.randn(batch_size, bag_size, 1)
    adjacency_matrix = torch.randn(batch_size, bag_size, bag_size)
    approx_sm = ApproxSm(alpha=0.5, num_steps=5)
    result = approx_sm(features, adjacency_matrix)
    assert result.shape == features.shape

def test_exactsm_solve_system(batch_size, bag_size, in_dim):
    exact_sm = ExactSm(alpha=0.5)
    A = torch.randn(batch_size, bag_size, bag_size)
    b = torch.randn(batch_size, bag_size, in_dim)
    result = exact_sm._solve_system(A, b)
    assert result.shape == b.shape
