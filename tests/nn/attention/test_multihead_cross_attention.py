import pytest
import torch
from torch.nn.attention import SDPBackend

from torchmil.nn.attention.multihead_cross_attention import MultiheadCrossAttention  # Assuming your script is saved as your_module.py

SDP_BACKEND = [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION,
               SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]

class TestMultiheadCrossAttention:
    @pytest.mark.parametrize("in_dim, out_dim, att_dim, n_heads, dropout, learn_weights",
                             [(10, 20, 32, 2, 0.1, True),
                              (16, 16, 64, 4, 0.0, False),
                              (8, None, 128, 8, 0.2, True),
                              (10, 5, 32, 2, 0.1, True),  # out_dim < att_dim
                              (10, 20, 32, 2, 0.1, False)]) # learn_weights = False
    def test_forward_pass(self, in_dim, out_dim, att_dim, n_heads, dropout, learn_weights):
        sample_input_x = torch.randn(2, 5, in_dim)  # batch_size, seq_len_x, in_dim
        sample_input_y = torch.randn(2, 7, in_dim)  # batch_size, seq_len_y, in_dim
        if att_dim % n_heads != 0:
            with pytest.raises(AssertionError):
                MultiheadCrossAttention(in_dim, out_dim, att_dim, n_heads, dropout, learn_weights)
            return
        layer = MultiheadCrossAttention(in_dim, out_dim, att_dim, n_heads, dropout, learn_weights)
        output = layer(sample_input_x, sample_input_y)
        expected_out_dim = out_dim if out_dim is not None else in_dim
        assert output.shape == (2, 5, expected_out_dim)

    def test_invalid_n_heads(self):
        in_dim = 10
        sample_input_x = torch.randn(2, 5, in_dim)  # batch_size, seq_len_x, in_dim
        sample_input_y = torch.randn(2, 7, in_dim)  # batch_size, seq_len_y, in_dim
        with pytest.raises(AssertionError):
            MultiheadCrossAttention(in_dim=10, att_dim=32, n_heads=3) # att_dim not divisible by n_heads

    def test_no_weights(self):
        in_dim = 10
        out_dim = 20
        att_dim = 32
        n_heads = 2
        sample_input_x = torch.randn(2, 5, in_dim)  # batch_size, seq_len_x, in_dim
        sample_input_y = torch.randn(2, 7, in_dim)  # batch_size, seq_len_y, in_dim
        if att_dim % n_heads != 0:
            pytest.skip("att_dim must be divisible by n_heads")
        layer = MultiheadCrossAttention(in_dim, out_dim, att_dim, n_heads, learn_weights=False)
        output = layer(sample_input_x, sample_input_y)
        assert output.shape == (2, 5, out_dim)

    def test_different_seq_len(self):
        in_dim = 10
        out_dim = 20
        att_dim = 32
        n_heads = 2
        if att_dim % n_heads != 0:
            pytest.skip("att_dim must be divisible by n_heads")
        layer = MultiheadCrossAttention(in_dim, out_dim, att_dim, n_heads, learn_weights=True)
        x1 = torch.randn(2, 5, in_dim)
        y1 = torch.randn(2, 7, in_dim)
        output1 = layer(x1, y1)
        assert output1.shape == (2, 5, out_dim)

        x2 = torch.randn(2, 8, in_dim)
        y2 = torch.randn(2, 3, in_dim)
        output2 = layer(x2, y2)
        assert output2.shape == (2, 8, out_dim)
