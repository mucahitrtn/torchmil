import pytest
import torch
from torchmil.nn.transformers.encoder import Encoder  # Assuming your script is saved as your_module.py

# Dummy Layer class for testing the Encoder with a simple layer
class DummyLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, X, return_att=False, **kwargs):
        Y = self.linear(X)
        if return_att:
            batch_size, seq_len, _ = X.shape
            attention_weights = torch.rand(batch_size, seq_len, seq_len)  # Dummy attention weights
            return Y, attention_weights
        return Y

@pytest.fixture
def sample_input():
    return torch.randn(2, 5, 10)  # batch_size, bag_size, in_dim

@pytest.fixture
def layers():
    # Create a list of DummyLayer instances
    return torch.nn.ModuleList([
        DummyLayer(10, 10),  # Layer 1: in_dim=10, out_dim=10
        DummyLayer(10, 10),  # Layer 2: in_dim=10, out_dim=10
        DummyLayer(10, 10)   # Layer 3: in_dim=10, out_dim=10
    ])

def test_encoder_forward_pass(sample_input, layers):
    encoder = Encoder(layers=layers, add_self=False)
    output = encoder(sample_input)
    assert output.shape == (2, 5, 10)  # Check the output shape

def test_encoder_forward_pass_add_self(sample_input, layers):
    encoder = Encoder(layers=layers, add_self=True)
    output = encoder(sample_input)
    assert output.shape == (2, 5, 10)  # Check the output shape

def test_encoder_forward_pass_return_att(sample_input, layers):
    encoder = Encoder(layers=layers, add_self=False)
    output, attention = encoder(sample_input, return_att=True)
    assert output.shape == (2, 5, 10)
    n_layers = len(layers)
    assert attention.shape == (n_layers, 2, 5, 5)

def test_encoder_with_different_layer_dimensions(sample_input, layers):
    #  Ensure that the dimensions of the layers are compatible with add_self
    layers = torch.nn.ModuleList([
        DummyLayer(10, 10),
        DummyLayer(10, 10),
        DummyLayer(10, 10)
    ])
    encoder = Encoder(layers=layers, add_self=False)
    output = encoder(sample_input)
    assert output.shape == (2, 5, 10)

def test_encoder_no_layers(sample_input):
    # Test the encoder with an empty list of layers
    encoder = Encoder(layers=torch.nn.ModuleList(), add_self=False)
    output = encoder(sample_input)
    assert output.shape == (2, 5, 10)  # Output should be the same as input
