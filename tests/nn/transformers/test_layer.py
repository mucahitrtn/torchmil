import pytest
import torch
from torchmil.nn.transformers.layer import Layer  # Assuming your script is saved as your_module.py

class DummyAttention(torch.nn.Module):
    def __init__(self, att_in_dim, att_out_dim):
        super().__init__()
        self.att_in_dim = att_in_dim
        self.att_out_dim = att_out_dim
        self.linear = torch.nn.Linear(att_in_dim, att_out_dim)

    def forward(self, x, return_att=False, **kwargs):
        out = self.linear(x)
        if return_att:
            # Return a dummy attention tensor
            batch_size, seq_len, _ = x.shape
            attention_weights = torch.rand(batch_size, seq_len, seq_len)
            return out, attention_weights
        return out

class DummyMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_dim, hidden_dim)
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))

@pytest.fixture
def sample_input():
    return torch.randn(2, 5, 50)  # batch_size, seq_len, in_dim

@pytest.mark.parametrize("in_dim, att_in_dim, att_out_dim, out_dim, use_mlp",
                         [
                             (10, 20, 20, 10, True),
                             (15, 15, 30, 30, False),
                             (8, 12, 12, 8, True),
                             (20, 10, 10, 20, False),
                             (10, 10, 15, 15, True),
                             # Test with None att_out_dim
                             (10, 20, None, 10, True),
                             (10, 20, 20, None, True)
                         ]
                         )  # Test with None out_dim
def test_layer_forward_pass(sample_input, in_dim, att_in_dim, att_out_dim, out_dim, use_mlp):
    # Handle None values for att_out_dim and out_dim in the test
    if att_out_dim is None:
        tmp_att_out_dim = att_in_dim
    else:
        tmp_att_out_dim = att_out_dim
    attention_module = DummyAttention(att_in_dim, tmp_att_out_dim)
    mlp_module = DummyMLP(tmp_att_out_dim, 4 * tmp_att_out_dim, tmp_att_out_dim, 0.1) if use_mlp else None
    expected_out_dim = out_dim if out_dim is not None else in_dim
    layer = Layer(att_module=attention_module, in_dim=in_dim, att_in_dim=att_in_dim, out_dim=out_dim, att_out_dim=att_out_dim, use_mlp=use_mlp, mlp_module=mlp_module if use_mlp else None)
    output = layer(sample_input[:, :, :in_dim])
    assert output.shape == (sample_input.shape[0], sample_input.shape[1], expected_out_dim)

@pytest.mark.parametrize("in_dim, att_in_dim, att_out_dim", [(10, 20, 20), (15, 15, 30)])
def test_layer_forward_with_return_attention(sample_input, in_dim, att_in_dim, att_out_dim):
    attention_module = DummyAttention(att_in_dim, att_out_dim)
    layer = Layer(att_module=attention_module, in_dim=in_dim, att_in_dim=att_in_dim, att_out_dim=att_out_dim)
    output, attention = layer(sample_input[:, :, :in_dim], return_att=True)
    expected_out_dim = in_dim
    assert output.shape == (sample_input.shape[0], sample_input.shape[1], expected_out_dim)
    assert attention.shape == (sample_input.shape[0], sample_input.shape[1], sample_input.shape[1])

@pytest.mark.parametrize("in_dim, att_in_dim, att_out_dim, out_dim", [(10, 10, 10, 10), (15, 15, 15, 30)])
def test_layer_no_projection(sample_input, in_dim, att_in_dim, att_out_dim, out_dim):
    attention_module = DummyAttention(att_in_dim, att_out_dim)
    layer = Layer(att_module=attention_module, in_dim=in_dim, att_in_dim=att_in_dim, att_out_dim=att_out_dim, out_dim=out_dim)
    output = layer(sample_input[:, :, :in_dim])
    assert output.shape == (sample_input.shape[0], sample_input.shape[1], out_dim)

def test_layer_with_custom_mlp(sample_input):
    in_dim = 10
    att_in_dim = 20
    att_out_dim = 25
    out_dim = 15
    custom_mlp = torch.nn.Linear(att_out_dim, att_out_dim)
    attention = DummyAttention(att_in_dim, att_out_dim)
    layer = Layer(att_module=attention, in_dim=in_dim, att_in_dim=att_in_dim, out_dim=out_dim, att_out_dim=att_out_dim, mlp_module=custom_mlp)
    output = layer(sample_input[:, :, :in_dim])
    assert output.shape == (sample_input.shape[0], sample_input.shape[1], out_dim)
