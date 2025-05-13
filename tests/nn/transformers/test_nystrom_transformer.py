import pytest
import torch
from torch.nn.attention import SDPBackend

from torchmil.nn.transformers.nystrom_transformer import NystromTransformerLayer, NystromTransformerEncoder

SDP_BACKEND = [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION,
               SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]


class TestNystromTransformerLayer:
    @pytest.mark.parametrize("in_dim, out_dim, att_dim, n_heads, learn_weights, n_landmarks, pinv_iterations, dropout, use_mlp",
                             [
                                (64, 128, 256, 4, True, 32, 3, 0.1, True),
                                (32, 32, 64, 2, False, 16, 6, 0.0, False),
                                (128, None, 128, 8, True, 64, 1, 0.2, True)
                            ])
    def test_forward_pass(self, in_dim, out_dim, att_dim, n_heads, learn_weights, n_landmarks, pinv_iterations, dropout, use_mlp):
        sample_input = torch.randn(2, 5, in_dim)
        layer = NystromTransformerLayer(in_dim, out_dim, att_dim, n_heads, learn_weights, n_landmarks, pinv_iterations, dropout, use_mlp)
        output = layer(sample_input)
        expected_out_dim = out_dim if out_dim is not None else in_dim
        assert output.shape == (2, 5, expected_out_dim)

    def test_forward_pass_with_mask(self):
        in_dim = 32
        att_dim = 64
        sample_input = torch.randn(2, 5, in_dim)
        sample_mask = torch.randint(0, 2, (2, 5)).bool()  # batch_size, bag_size
        layer = NystromTransformerLayer(in_dim, att_dim=att_dim)
        output = layer(sample_input, mask=sample_mask)
        assert output.shape == (2, 5, in_dim)

    def test_forward_pass_return_att(self):
        in_dim = 32
        att_dim = 64
        n_heads = 2
        sample_input = torch.randn(2, 5, in_dim)
        layer = NystromTransformerLayer(in_dim, att_dim=att_dim, n_heads=n_heads)
        output, attention = layer(sample_input, return_att=True)
        assert output.shape == (2, 5, in_dim)
        assert attention.shape == (2, n_heads, 5, 5)  # batch_size, n_heads, bag_size, bag_size


class TestNystromTransformerEncoder:
    @pytest.mark.parametrize("in_dim, out_dim, att_dim, n_heads, n_layers, n_landmarks, pinv_iterations, dropout, use_mlp, add_self",
                             [(32, 64, 128, 2, 2, 16, 3, 0.1, True, False),
                              (64, 64, 64, 4, 1, 32, 6, 0.0, False, True),
                              (128, None, 256, 8, 3, 64, 1, 0.2, True, False)])
    def test_forward_pass(self, in_dim, out_dim, att_dim, n_heads, n_layers, n_landmarks, pinv_iterations, dropout, use_mlp, add_self):
        sample_input = torch.randn(2, 5, in_dim)
        encoder = NystromTransformerEncoder(in_dim, out_dim, att_dim, n_heads, n_layers, n_landmarks, pinv_iterations, dropout, use_mlp, add_self)
        output = encoder(sample_input)
        expected_out_dim = out_dim if out_dim is not None else in_dim
        assert output.shape == (2, 5, expected_out_dim)

    def test_forward_pass_with_mask(self):
        in_dim = 32
        att_dim = 64
        n_layers = 2
        sample_input = torch.randn(2, 5, in_dim)
        sample_mask = torch.randint(0, 2, (2, 5)).bool()  # batch_size, bag_size
        encoder = NystromTransformerEncoder(in_dim, att_dim=att_dim, n_layers=n_layers)
        output = encoder(sample_input, mask=sample_mask)
        assert output.shape == (2, 5, in_dim)

    def test_forward_pass_return_att(self):
        in_dim = 32
        att_dim = 64
        n_heads = 2
        n_layers = 2
        sample_input = torch.randn(2, 5, in_dim)
        encoder = NystromTransformerEncoder(in_dim, att_dim=att_dim, n_heads=n_heads, n_layers=n_layers)
        output, attention = encoder(sample_input, return_att=True)
        assert output.shape == (2, 5, in_dim)
        assert attention[0].shape == (2, n_heads, 5, 5)  # batch_size, n_heads, bag_size, bag_size

    def test_add_self_true(self):
        in_dim = 32
        att_dim = 32
        n_layers = 1
        sample_input = torch.randn(2, 5, in_dim)
        encoder = NystromTransformerEncoder(in_dim, att_dim=att_dim, n_heads=2, n_layers=n_layers, add_self=True)
        output = encoder(sample_input)
        assert output.shape == (2, 5, in_dim)  # Output should have same shape as input

    def test_encoder_no_layers(self):
        in_dim = 32
        att_dim = 64
        sample_input = torch.randn(2, 5, in_dim)
        encoder = NystromTransformerEncoder(in_dim=in_dim, att_dim=att_dim, n_layers=0, add_self=False)
        output = encoder(sample_input)
        assert output.shape == (2, 5, in_dim)