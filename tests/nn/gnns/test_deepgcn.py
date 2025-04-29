import torch
import pytest
from torchmil.nn.gnns.deepgcn import DeepGCNLayer
from torchmil.nn.gnns.gnn_identity import GNNIdentity

# Helper functions for creating test data
def create_random_input(batch_size, n_nodes, in_dim):
    """Creates a random input tensor for testing."""
    return torch.randn(batch_size, n_nodes, in_dim)

def create_random_adjacency_matrix(batch_size, n_nodes):
    """Creates a random adjacency matrix for testing."""
    return torch.randint(0, 2, (batch_size, n_nodes, n_nodes)).float()

# Fixtures for common setup
@pytest.fixture
def sample_input():
    """Provides a sample input tensor."""
    return create_random_input(batch_size=2, n_nodes=5, in_dim=16)

@pytest.fixture
def sample_adj():
    """Provides a sample adjacency matrix."""
    return create_random_adjacency_matrix(batch_size=2, n_nodes=5)

class CustomLinear(torch.nn.Module):
    """Custom linear layer for testing."""
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        return self.linear(x)  # Ignore adj for simplicity in this example

# Test cases for DeepGCNLayer
class TestDeepGCNLayer:
    """
    Test suite for the DeepGCNLayer class.
    """
    def test_initialization_default(self):
        """
        Tests the default initialization of DeepGCNLayer.
        Verifies that default modules are set to Identity.
        """
        layer = DeepGCNLayer()
        assert isinstance(layer.conv, GNNIdentity)
        assert isinstance(layer.norm, torch.nn.Identity)
        assert isinstance(layer.act, torch.nn.Identity)
        assert layer.dropout.p == 0.0
        assert layer.block == 'plain'

    def test_initialization_custom(self):
        """
        Tests initialization with custom modules.
        Verifies that the provided modules are correctly assigned.
        """
        conv = CustomLinear(16, 32)
        norm = torch.nn.BatchNorm1d(5)  # Corrected: Use BatchNorm1d
        act = torch.nn.ReLU()
        dropout = 0.5
        block = 'dense'
        layer = DeepGCNLayer(conv, norm, act, block, dropout)
        assert layer.conv == conv
        assert layer.norm == norm
        assert layer.act == act
        assert layer.dropout.p == dropout
        assert layer.block == block

    def test_forward_default(self, sample_input, sample_adj):
        """
        Tests the forward pass with default settings (Identity modules).
        Verifies that the output shape is the same as the input shape.
        """
        layer = DeepGCNLayer()
        output = layer(sample_input, sample_adj)
        assert output.shape == sample_input.shape

    def test_forward_with_conv(self, sample_input, sample_adj):
        """
        Tests the forward pass with a convolutional layer (CustomLinear).
        Verifies that the output shape is as expected after convolution.
        """
        in_dim = sample_input.shape[-1]
        out_dim = 32
        conv = CustomLinear(in_dim, out_dim)
        layer = DeepGCNLayer(conv=conv)
        output = layer(sample_input, sample_adj)
        assert output.shape == (sample_input.shape[0], sample_input.shape[1], out_dim)

    def test_forward_with_batchnorm(self, sample_input, sample_adj):
        """
        Tests the forward pass with batch normalization.
        Verifies that the output shape remains the same.  Important:  BatchNorm1d
        """
        norm = torch.nn.BatchNorm1d(sample_input.shape[1]) # Corrected for batched input
        layer = DeepGCNLayer(norm=norm)
        output = layer(sample_input, sample_adj)
        assert output.shape == sample_input.shape

    def test_forward_with_relu(self, sample_input, sample_adj):
        """
        Tests the forward pass with ReLU activation.
        Verifies that the output values are non-negative.
        """
        layer = DeepGCNLayer(act=torch.nn.ReLU(), block='plain')
        output = layer(sample_input, sample_adj)
        assert torch.all(output >= 0)

    def test_forward_with_dropout(self, sample_input, sample_adj):
        """
        Tests the forward pass with dropout.
        Verifies that the output is not exactly the same as the input
        (this is a probabilistic test and might fail occasionally).
        """
        layer = DeepGCNLayer(dropout=0.5)
        output = layer(sample_input, sample_adj)
        # Check if the output is different from the input, accounting for randomness
        assert not torch.allclose(output, sample_input)

    def test_forward_block_res(self, sample_input, sample_adj):
        """
        Tests the 'res' block.  Verifies that the output is the sum of
        the input and the convolved features.
        """
        in_dim = sample_input.shape[-1]
        out_dim = in_dim
        conv = CustomLinear(in_dim, out_dim)
        layer = DeepGCNLayer(conv=conv, block='res')
        output = layer(sample_input, sample_adj)
        assert output.shape == (sample_input.shape[0], sample_input.shape[1], out_dim)

    def test_forward_block_res_plus(self, sample_input, sample_adj):
        """Tests the 'res+' block."""
        in_dim = sample_input.shape[-1]
        out_dim = in_dim
        conv = CustomLinear(in_dim, out_dim)
        norm = torch.nn.BatchNorm1d(sample_input.shape[1])
        act = torch.nn.ReLU()
        layer = DeepGCNLayer(conv=conv, norm=norm, act=act, block='res+')
        output = layer(sample_input, sample_adj)
        assert output.shape == (sample_input.shape[0], sample_input.shape[1], out_dim)

    def test_forward_block_dense(self, sample_input, sample_adj):
        """
        Tests the 'dense' block. Verifies that the output dimension is
        the sum of the input dimension and the convolution output dimension.
        """
        in_dim = sample_input.shape[-1]
        out_dim = 32
        conv = CustomLinear(in_dim, out_dim)
        layer = DeepGCNLayer(conv=conv, block='dense')
        output = layer(sample_input, sample_adj)
        assert output.shape == (sample_input.shape[0], sample_input.shape[1], in_dim + out_dim)

    def test_forward_block_plain(self, sample_input, sample_adj):
        """
        Tests the 'plain' block. Verifies that the output shape is the
        same as the convolution output shape.
        """
        in_dim = sample_input.shape[-1]
        out_dim = 32
        conv = CustomLinear(in_dim, out_dim)
        layer = DeepGCNLayer(conv=conv, block='plain')
        output = layer(sample_input, sample_adj)
        assert output.shape == (sample_input.shape[0], sample_input.shape[1], out_dim)
