import torch
import pytest
from torchmil.nn.gnns.gnn_identity import GNNIdentity # change your_module

# Fixture for creating a GNNIdentity instance
@pytest.fixture
def gnn_identity_instance():
    return GNNIdentity()

# Test case 1: Check the output shape and value when adj is None
def test_forward_no_adj(gnn_identity_instance):
    batch_size = 2
    n_nodes = 5
    in_dim = 10
    x = torch.randn(batch_size, n_nodes, in_dim)
    
    output = gnn_identity_instance(x)
    
    assert torch.allclose(output, x), "Output should be identical to input when adj is None"
    assert output.shape == x.shape, "Output shape should match input shape"

# Test case 2: Check the output shape and value when adj is not None
def test_forward_with_adj(gnn_identity_instance):
    batch_size = 3
    n_nodes = 4
    in_dim = 8
    x = torch.randn(batch_size, n_nodes, in_dim)
    adj = torch.randn(batch_size, n_nodes, n_nodes)  # Adjacency matrix
    
    output = gnn_identity_instance(x, adj)
    
    assert torch.allclose(output, x), "Output should be identical to input even when adj is not None"
    assert output.shape == x.shape, "Output shape should match input shape"

# Test case 3: Check with different input shapes
def test_forward_different_shapes(gnn_identity_instance):
    shapes = [(1, 1, 1), (2, 3, 4), (5, 2, 7), (10, 10, 10)]
    for shape in shapes:
        x = torch.randn(shape)
        output = gnn_identity_instance(x)
        assert torch.allclose(output, x), f"Output should be identical to input for shape {shape}"
        assert output.shape == shape, f"Output shape should match input shape for {shape}"

