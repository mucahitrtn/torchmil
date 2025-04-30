import torch
import pytest

from torchmil.nn.gnns.dense_mincut_pool import trace, diag, dense_mincut_pool

# Helper Functions
def create_random_tensor(shape, requires_grad=False):
    """Creates a random tensor with specified shape."""
    return torch.randn(shape, requires_grad=requires_grad)

def create_random_int_tensor(shape, low=0, high=1):
    """Creates a random integer tensor."""
    return torch.randint(low, high, shape).float()

# Test cases for trace function
def test_trace_rank3():
    """Tests the trace function with a rank-3 tensor."""
    x = create_random_tensor((2, 3, 3))
    result = trace(x)
    assert result.shape == (2,)
    expected_trace = torch.tensor([torch.trace(x[i]) for i in range(x.shape[0])])
    assert torch.allclose(result, expected_trace)

def test_trace_rank3_non_square():
    """Tests the trace function with a non-square rank-3 tensor. Should raise error"""
    x = create_random_tensor((2, 3, 4))
    with pytest.raises(RuntimeError):  # Expecting a RuntimeError for non-square matrices
        trace(x)

# Test cases for diag function
def test_diag_rank2():
    """Tests the diag function with a rank-2 tensor."""
    x = create_random_tensor((2, 3))
    result = diag(x)
    assert result.shape == (2, 3, 3)
    expected_diag = torch.zeros((2, 3, 3))
    for i in range(2):
        for j in range(3):
            expected_diag[i, j, j] = x[i, j]
    assert torch.allclose(result, expected_diag)

def test_diag_rank2_sparse():
    """Tests the diag function with a sparse rank-2 tensor."""
    x = create_random_tensor((2, 3)).to_sparse()
    result = diag(x)
    assert result.shape == (2, 3, 3)
    expected_diag = torch.zeros((2, 3, 3))
    x_dense = x.to_dense()
    for i in range(2):
        for j in range(3):
            expected_diag[i, j, j] = x_dense[i, j]
    assert torch.allclose(result, expected_diag)

# Test cases for dense_mincut_pool function
def test_dense_mincut_pool_shapes():
    """Tests the output shapes of dense_mincut_pool."""
    batch_size = 2
    n_nodes = 5
    in_dim = 16
    n_cluster = 3
    x = create_random_tensor((batch_size, n_nodes, in_dim))
    adj = create_random_tensor((batch_size, n_nodes, n_nodes))
    s = create_random_tensor((batch_size, n_nodes, n_cluster))
    x_, adj_, mincut_loss, ortho_loss = dense_mincut_pool(x, adj, s)
    assert x_.shape == (batch_size, n_cluster, in_dim)
    assert adj_.shape == (batch_size, n_cluster, n_cluster)
    assert mincut_loss.shape == ()  # Scalar loss
    assert ortho_loss.shape == ()   # Scalar loss

def test_dense_mincut_pool_mask():
    """Tests dense_mincut_pool with a mask."""
    batch_size = 2
    n_nodes = 5
    in_dim = 16
    n_cluster = 3
    x = create_random_tensor((batch_size, n_nodes, in_dim))
    adj = create_random_tensor((batch_size, n_nodes, n_nodes))
    s = create_random_tensor((batch_size, n_nodes, n_cluster))
    mask = create_random_int_tensor((batch_size, n_nodes))
    x_, adj_, mincut_loss, ortho_loss = dense_mincut_pool(x, adj, s, mask=mask)
    assert x_.shape == (batch_size, n_cluster, in_dim)
    assert adj_.shape == (batch_size, n_cluster, n_cluster)
    assert mincut_loss.shape == ()
    assert ortho_loss.shape == ()

def test_dense_mincut_pool_temp():
    """Tests dense_mincut_pool with different temperature values."""
    batch_size = 2
    n_nodes = 5
    in_dim = 16
    n_cluster = 3
    x = create_random_tensor((batch_size, n_nodes, in_dim))
    adj = create_random_tensor((batch_size, n_nodes, n_nodes))
    s = create_random_tensor((batch_size, n_nodes, n_cluster))
    temp1 = 1.0
    temp2 = 0.5
    x_1, adj_1, mincut_loss_1, ortho_loss_1 = dense_mincut_pool(x, adj, s, temp=temp1)
    x_2, adj_2, mincut_loss_2, ortho_loss_2 = dense_mincut_pool(x, adj, s, temp=temp2)
    assert x_1.shape == x_2.shape
    assert adj_1.shape == adj_2.shape
    assert mincut_loss_1.shape == mincut_loss_2.shape
    assert ortho_loss_1.shape == ortho_loss_2.shape

def test_dense_mincut_pool_2d_input():
    """Tests dense_mincut_pool with 2D input tensors."""
    n_nodes = 5
    in_dim = 16
    n_cluster = 3
    x = create_random_tensor((n_nodes, in_dim))  # Shape: (n_nodes, in_dim)
    adj = create_random_tensor((n_nodes, n_nodes))  # Shape: (n_nodes, n_nodes)
    s = create_random_tensor((n_nodes, n_cluster))  # Shape: (n_nodes, n_cluster)
    x_, adj_, mincut_loss, ortho_loss = dense_mincut_pool(x, adj, s)
    assert x_.shape == (1, n_cluster, in_dim)
    assert adj_.shape == (1, n_cluster, n_cluster)
    assert mincut_loss.shape == ()
    assert ortho_loss.shape == ()
