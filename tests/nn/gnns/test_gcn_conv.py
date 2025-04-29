import torch
import pytest
from torchmil.nn.gnns.gcn_conv import GCNConv

# Test case 1: Check the output shape of the forward pass.
def test_forward_output_shape():
    batch_size = 2
    n_nodes = 5
    in_dim = 10
    out_dim = 20
    gcn_conv = GCNConv(in_dim, out_dim, learn_weights=True)

    x = torch.randn(batch_size, n_nodes, in_dim)
    adj = torch.randn(batch_size, n_nodes, n_nodes)
    
    output = gcn_conv(x, adj)
    
    assert output.shape == (batch_size, n_nodes, out_dim), "Output shape is incorrect"

# Test case 2: Check if self-loops are added correctly when add_self_loops is True.
def test_add_self_loops():
    in_dim = 5
    out_dim = 5
    add_self_loops = True
    gcn_conv = GCNConv(in_dim, out_dim, add_self_loops=add_self_loops)
    
    batch_size = 1
    n_nodes = 3
    x = torch.randn(batch_size, n_nodes, in_dim)
    adj = torch.eye(n_nodes).unsqueeze(0)  # Identity matrix as adjacency
    
    output = gcn_conv(x, adj)
    
    # The key idea:  If self-loops are added correctly, the output should be different
    # from just multiplying by the original adjacency matrix.  We check if the output
    # is element-wise close to the input, which would happen with identity matrix + self loops
    assert not torch.allclose(output, torch.bmm(adj, x)), "Self-loops not added correctly"
    
def test_no_self_loops():
    in_dim = 5
    out_dim = 5
    add_self_loops = False
    gcn_conv = GCNConv(in_dim, out_dim, add_self_loops=add_self_loops)
    
    batch_size = 1
    n_nodes = 3
    x = torch.randn(batch_size, n_nodes, in_dim)
    adj = torch.eye(n_nodes).unsqueeze(0)
    
    output = gcn_conv(x, adj)
    assert torch.allclose(output, torch.bmm(adj, x)), "Self-loops incorrectly added when add_self_loops is False"

# Test case 3: Check if the output is normalized when normalize is True.
def test_normalize():
    in_dim = 4
    out_dim = 3
    normalize = True
    gcn_conv = GCNConv(in_dim, out_dim, normalize=normalize)
    
    batch_size = 1
    n_nodes = 2
    x = torch.randn(batch_size, n_nodes, in_dim)
    adj = torch.randn(batch_size, n_nodes, n_nodes)
    
    output = gcn_conv(x, adj)
    
    # Check if the norm of each node feature vector is approximately 1
    norms = torch.norm(output, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Output is not normalized"

# Test case 4: Check if layer normalization is applied when layer_norm is True.
def test_layer_norm():
    in_dim = 6
    out_dim = 6
    layer_norm = True
    gcn_conv = GCNConv(in_dim, out_dim, layer_norm=layer_norm)
    
    batch_size = 1
    n_nodes = 2
    x = torch.randn(batch_size, n_nodes, in_dim)
    adj = torch.randn(batch_size, n_nodes, n_nodes)

    layer_norm = torch.nn.LayerNorm(out_dim)
    
    output = gcn_conv(x, adj)

    y = torch.bmm(adj, x)
    y = layer_norm(y)
    
    assert torch.allclose(output, y), "Layer normalization not applied correctly"

# Test case 5: Test for the case where learn_weights is True
def test_learn_weights():
    in_dim = 7
    out_dim = 8
    learn_weights = True
    gcn_conv = GCNConv(in_dim, out_dim, learn_weights=learn_weights)
    
    batch_size = 1
    n_nodes = 2
    x = torch.randn(batch_size, n_nodes, in_dim)
    adj = torch.randn(batch_size, n_nodes, n_nodes)
    
    output = gcn_conv(x, adj)
    assert output.shape == (batch_size, n_nodes, out_dim)

# Test case 6: Test for the case where learn_weights is False
def test_no_learn_weights():
    in_dim = 9
    out_dim = 9
    learn_weights = False
    gcn_conv = GCNConv(in_dim, out_dim, learn_weights=learn_weights)
    
    batch_size = 1
    n_nodes = 2
    x = torch.randn(batch_size, n_nodes, in_dim)
    adj = torch.randn(batch_size, n_nodes, n_nodes)
    
    output = gcn_conv(x, adj)
    assert torch.allclose(output, torch.bmm(adj, x)), "Output should be same as input when learn_weights is False"

# Test case 7: Test different activation functions
def test_activation_function():
    in_dim = 5
    out_dim = 5
    activations = [torch.nn.ReLU(), torch.nn.Tanh(), torch.nn.Sigmoid()]
    for activation in activations:
        gcn_conv = GCNConv(in_dim, out_dim, activation=activation)
        batch_size = 1
        n_nodes = 2
        x = torch.randn(batch_size, n_nodes, in_dim)
        adj = torch.randn(batch_size, n_nodes, n_nodes)
        output = gcn_conv(x, adj)
        assert output.shape == (batch_size, n_nodes, out_dim) # check shape
        # check that the output is different from the input, implying the activation
        # function was applied
        assert not torch.allclose(output, torch.bmm(adj, x))

# Test case 8: Test different dropout values
def test_dropout_function():
    in_dim = 5
    out_dim = 5
    dropouts = [0.0, 0.2, 0.5, 0.8]
    for dropout in dropouts:
        gcn_conv = GCNConv(in_dim, out_dim, dropout=dropout)
        batch_size = 1
        n_nodes = 2
        x = torch.randn(batch_size, n_nodes, in_dim)
        adj = torch.randn(batch_size, n_nodes, n_nodes)
        output = gcn_conv(x, adj)
        assert output.shape == (batch_size, n_nodes, out_dim) # check shape

# Test case 9: Test with and without bias
def test_bias():
    in_dim = 5
    out_dim = 5
    bias_options = [True, False]
    for bias in bias_options:
        gcn_conv = GCNConv(in_dim, out_dim, bias=bias)
        batch_size = 1
        n_nodes = 2
        x = torch.randn(batch_size, n_nodes, in_dim)
        adj = torch.randn(batch_size, n_nodes, n_nodes)
        output = gcn_conv(x, adj)
        assert output.shape == (batch_size, n_nodes, out_dim)

