import pytest
import torch

from torchmil.nn.attention.sm_attention_pool import SmAttentionPool

@pytest.mark.parametrize("batch_size, bag_size, in_dim", [(2, 10, 10), (4, 20, 20), (1, 5, 30)])
def test_sm_attention_pool_forward(batch_size, bag_size, in_dim):
    """
    Test the forward pass of the SmAttentionPool module.
    """
    pool = SmAttentionPool(in_dim)
    X = torch.randn(batch_size, bag_size, in_dim)
    adj = torch.eye(bag_size).unsqueeze(0).repeat(batch_size, 1, 1)  # Identity adjacency matrix
    z = pool(X, adj)
    assert z.shape == (batch_size, in_dim)

def test_sm_attention_pool_forward_with_mask():
    """
    Test the forward pass of the SmAttentionPool module with a mask.
    """
    batch_size = 2
    bag_size = 10
    in_dim = 10
    pool = SmAttentionPool(in_dim)
    X = torch.randn(batch_size, bag_size, in_dim)
    adj = torch.eye(bag_size).unsqueeze(0).repeat(batch_size, 1, 1)
    mask = torch.randint(0, 2, (batch_size, bag_size)).bool()
    z = pool(X, adj, mask=mask)
    assert z.shape == (batch_size, in_dim)

def test_sm_attention_pool_forward_return_att():
    """
    Test the forward pass of the SmAttentionPool module with return_att=True.
    """
    batch_size = 2
    bag_size = 10
    in_dim = 10
    pool = SmAttentionPool(in_dim)
    X = torch.randn(batch_size, bag_size, in_dim)
    adj = torch.eye(bag_size).unsqueeze(0).repeat(batch_size, 1, 1)
    z, att = pool(X, adj, return_att=True)
    assert z.shape == (batch_size, in_dim)
    assert att.shape == (batch_size, bag_size)

def test_sm_attention_pool_invalid_act():
    """
    Test that an error is raised when an invalid activation function is provided.
    """
    with pytest.raises(ValueError):
        SmAttentionPool(10, act="invalid")
