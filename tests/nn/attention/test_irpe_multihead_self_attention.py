import pytest
import torch

from torchmil.nn.attention.irpe_multihead_self_attention import iRPEMultiheadSelfAttention


class TestiRPEMultiheadSelfAttention:
    @pytest.mark.parametrize(
        "in_dim, out_dim, att_dim, n_heads, dropout, learn_weights, rpe_ratio, rpe_method, rpe_mode, rpe_shared_head, rpe_skip, rpe_on",
        [
            (10, 20, 32, 2, 0.1, True, 2.0, "product", "contextual", True, 1, "k"),
            (16, 16, 64, 4, 0.0, False, 1.5, "cross", "bias", False, 0, "q"),
            (8, None, 128, 8, 0.2, True, 1.0, "product", "contextual", True, 1, "v"),
            (10, 5, 32, 2, 0.1, True, 2.0, "cross", "bias", False, 1, "k"),  # out_dim < att_dim
            (10, 20, 32, 2, 0.1, False, 1.5, "product", "contextual", True, 0, "q"), # learn_weights = False
        ],
    )
    def test_forward_pass(
        self,
        in_dim,
        out_dim,
        att_dim,
        n_heads,
        dropout,
        learn_weights,
        rpe_ratio,
        rpe_method,
        rpe_mode,
        rpe_shared_head,
        rpe_skip,
        rpe_on,
    ):
        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, in_dim)
        layer = iRPEMultiheadSelfAttention(
            in_dim,
            out_dim,
            att_dim,
            n_heads,
            dropout,
            learn_weights,
            rpe_ratio,
            rpe_method,
            rpe_mode,
            rpe_shared_head,
            rpe_skip,
            rpe_on,
        )
        output = layer(x)
        expected_out_dim = out_dim if out_dim is not None else in_dim
        assert output.shape == (batch_size, seq_len, expected_out_dim)

    def test_forward_pass_with_mask(self):
        in_dim = 10
        att_dim = 32
        n_heads = 2
        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, in_dim)
        mask = torch.randint(0, 2, (batch_size, seq_len)).bool()
        layer = iRPEMultiheadSelfAttention(in_dim, att_dim=att_dim, n_heads=n_heads)
        output = layer(x, mask=mask)
        assert output.shape == (batch_size, seq_len, in_dim)

    def test_forward_pass_return_att(self):
        in_dim = 10
        att_dim = 32
        n_heads = 2
        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, in_dim)
        layer = iRPEMultiheadSelfAttention(in_dim, att_dim=att_dim, n_heads=n_heads)
        output, attention = layer(x, return_att=True)
        assert output.shape == (batch_size, seq_len, in_dim)
        assert attention.shape == (batch_size, n_heads, seq_len, seq_len)

    def test_invalid_n_heads(self):
        in_dim = 10
        att_dim = 32
        n_heads = 3
        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, in_dim)
        with pytest.raises(AssertionError):
            iRPEMultiheadSelfAttention(in_dim=in_dim, att_dim=att_dim, n_heads=n_heads)

    def test_no_weights(self):
        in_dim = 10
        out_dim = 20
        att_dim = 32
        n_heads = 2
        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, in_dim)
        layer = iRPEMultiheadSelfAttention(in_dim, out_dim, att_dim, n_heads, learn_weights=False)
        output = layer(x)
        assert output.shape == (batch_size, seq_len, out_dim)

    def test_height_width(self):
        in_dim = 10
        out_dim = 20
        att_dim = 32
        n_heads = 2
        batch_size = 2
        seq_len = 25  # height = 5, width = 5
        x = torch.randn(batch_size, seq_len, in_dim)
        layer = iRPEMultiheadSelfAttention(in_dim, out_dim, att_dim, n_heads, rpe_skip=0)
        output = layer(x, height=5, width=5)
        assert output.shape == (batch_size, seq_len, out_dim)

    def test_different_rpe_on(self):
        in_dim = 10
        out_dim = 20
        att_dim = 32
        n_heads = 2
        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, in_dim)
        for rpe_on in ['q', 'k', 'v']:
            layer = iRPEMultiheadSelfAttention(in_dim, out_dim, att_dim, n_heads, rpe_on=rpe_on)
            output = layer(x)
            assert output.shape == (batch_size, seq_len, out_dim)
