import pytest
import torch

from torchmil.nn.attention.nystrom_attention import NystromAttention

def create_random_input(batch_size, seq_len, in_dim):
    return torch.randn(batch_size, seq_len, in_dim)

def create_random_mask(batch_size, seq_len):
    return torch.randint(0, 2, (batch_size, seq_len)).bool()

class TestNystromAttention:
    @pytest.mark.parametrize("in_dim, out_dim, att_dim, n_heads, n_landmarks, learn_weights, return_att", [
        (32, 64, 128, 4, 32, True, False),
        (16, 16, 64, 2, 64, False, True),
        (24, 32, 96, 3, 128, True, False),
        (32, 64, 128, 4, 20, True, False), # n_landmarks < seq_len
        (32, 64, 128, 4, 256, True, True)
    ])
    def test_forward_pass(self, in_dim, out_dim, att_dim, n_heads, n_landmarks, learn_weights, return_att):
        batch_size = 2
        seq_len = 100
        x = create_random_input(batch_size, seq_len, in_dim)
        layer = NystromAttention(in_dim, out_dim, att_dim, n_heads, learn_weights, n_landmarks)
        output = layer(x, return_att=return_att)

        expected_out_dim = out_dim if out_dim is not None else in_dim
        if return_att:
            y, att = output
            assert y.shape == (batch_size, seq_len, expected_out_dim)
            assert att.shape == (batch_size, n_heads, seq_len, seq_len)
        else:
            y = output
            assert y.shape == (batch_size, seq_len, expected_out_dim)

    @pytest.mark.parametrize("in_dim, att_dim, n_heads, n_landmarks", [(32, 128, 4, 32)])
    def test_forward_pass_with_mask(self, in_dim, att_dim, n_heads, n_landmarks):
        batch_size = 2
        seq_len = 100
        x = create_random_input(batch_size, seq_len, in_dim)
        mask = create_random_mask(batch_size, seq_len)
        layer = NystromAttention(in_dim, att_dim=att_dim, n_heads=n_heads, n_landmarks=n_landmarks)
        output = layer(x, mask=mask)
        assert output.shape == (batch_size, seq_len, in_dim)

    @pytest.mark.parametrize("in_dim, att_dim, n_heads, n_landmarks", [(32, 128, 4, 32)])
    def test_forward_pass_variable_seq_len(self, in_dim, att_dim, n_heads, n_landmarks):
        batch_size = 2
        seq_len1 = 100
        seq_len2 = 120
        x1 = create_random_input(batch_size, seq_len1, in_dim)
        x2 = create_random_input(batch_size, seq_len2, in_dim)
        layer = NystromAttention(in_dim, att_dim=att_dim, n_heads=n_heads, n_landmarks=n_landmarks)
        output1 = layer(x1)
        output2 = layer(x2)
        assert output1.shape == (batch_size, seq_len1, in_dim)
        assert output2.shape == (batch_size, seq_len2, in_dim)

    def test_no_weights(self):
        in_dim = 32
        out_dim = 64
        att_dim = 128
        n_heads = 4
        n_landmarks = 32
        batch_size = 2
        seq_len = 100
        x = create_random_input(batch_size, seq_len, in_dim)
        layer = NystromAttention(in_dim, out_dim, att_dim, n_heads, learn_weights=False, n_landmarks=n_landmarks)
        output = layer(x)
        assert output.shape == (batch_size, seq_len, out_dim)
