import pytest
import torch
from torch.nn.attention import SDPBackend

from torchmil.nn.transformers.conventional_transformer import TransformerLayer, TransformerEncoder

SDP_BACKEND = [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION,
               SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]

class TestTransformerLayer:
    @pytest.mark.parametrize("in_dim, out_dim, att_dim, n_heads, use_mlp, dropout",
                             [(10, 20, 32, 2, True, 0.1),
                              (15, 15, 64, 4, False, 0.0),
                              (8, None, 128, 8, True, 0.2)])
    def test_forward_pass(self, in_dim, out_dim, att_dim, n_heads, use_mlp, dropout):
        sample_input = torch.randn(2, 5, in_dim)
        layer = TransformerLayer(in_dim, out_dim, att_dim, n_heads, use_mlp, dropout)
        output = layer(sample_input)
        expected_out_dim = out_dim if out_dim is not None else in_dim
        assert output.shape == (2, 5, expected_out_dim)

    def test_forward_pass_with_mask(self):
        in_dim = 10
        sample_input = torch.randn(2, 5, in_dim)
        sample_mask = torch.randint(0, 2, (2, 5)).bool()  # batch_size, bag_size
        layer = TransformerLayer(in_dim)
        output = layer(sample_input, mask=sample_mask)
        assert output.shape == (2, 5, 10)

    def test_forward_pass_return_att(self):
        in_dim = 10
        sample_input = torch.randn(2, 5, in_dim)
        layer = TransformerLayer(in_dim)
        output, attention = layer(sample_input, return_att=True)
        assert output.shape == (2, 5, 10)
        assert attention.shape == (2, 4, 5, 5)  # batch_size, n_heads, bag_size, bag_size


class TestTransformerEncoder:
    @pytest.mark.parametrize("in_dim, out_dim, att_dim, n_heads, n_layers, use_mlp, add_self, dropout",
                             [(10, 20, 32, 2, 2, True, False, 0.1),
                              (20, 20, 20, 4, 3, False, True, 0.0),
                              (8, None, 128, 8, 1, True, False, 0.2)])
    def test_forward_pass(self, in_dim, out_dim, att_dim, n_heads, n_layers, use_mlp, add_self, dropout):
        sample_input = torch.randn(2, 5, in_dim)
        encoder = TransformerEncoder(in_dim, out_dim, att_dim, n_heads, n_layers, use_mlp, add_self, dropout)
        output = encoder(sample_input)
        expected_out_dim = out_dim if out_dim is not None else in_dim
        assert output.shape == (2, 5, expected_out_dim)

    def test_forward_pass_with_mask(self):
        in_dim = 10
        sample_input = torch.randn(2, 5, in_dim)
        sample_mask = torch.randint(0, 2, (2, 5)).bool()  # batch_size, bag_size
        encoder = TransformerEncoder(in_dim, n_layers=2)
        output = encoder(sample_input, mask=sample_mask)
        assert output.shape == (2, 5, 10)

    def test_forward_pass_return_att(self):
        in_dim = 10
        sample_input = torch.randn(2, 5, in_dim)
        encoder = TransformerEncoder(in_dim, n_layers=2)
        output, attention = encoder(sample_input, return_att=True)
        assert output.shape == (2, 5, 10)
        assert attention.shape == (2, 2, 4, 5, 5)  # n_layers, batch_size, n_heads, bag_size, bag_size

    def test_add_self_true(self):
        in_dim = 10
        sample_input = torch.randn(2, 5, in_dim)
        encoder = TransformerEncoder(in_dim, att_dim=in_dim, n_heads=2, n_layers=2, add_self=True)
        output = encoder(sample_input)
        assert output.shape == (2, 5, 10)  # Output should have same shape as input

    def test_encoder_no_layers(self):
        sample_input = torch.randn(2, 5, 10)
        encoder = TransformerEncoder(in_dim=10, n_layers=0, add_self=False)
        output = encoder(sample_input)
        assert output.shape == (2, 5, 10)
