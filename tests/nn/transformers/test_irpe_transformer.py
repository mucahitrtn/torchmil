import pytest
import torch
from torch.nn.attention import SDPBackend

from torchmil.nn.transformers.irpe_transformer import iRPETransformerLayer, iRPETransformerEncoder

SDP_BACKEND = [
    SDPBackend.MATH,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
]


class TestiRPETransformerLayer:
    @pytest.mark.parametrize(
        "in_dim, out_dim, att_dim, n_heads, use_mlp, dropout, rpe_ratio, rpe_method, rpe_mode, rpe_shared_head, rpe_skip, rpe_on",
        [
            (64, 128, 256, 4, True, 0.1, 2.0, "product", "contextual", True, 1, "k"),
            (32, 32, 64, 2, False, 0.0, 1.5, "quant", "bias", False, 0, "q"),
            (128, None, 128, 8, True, 0.2, 1.0, "cross", "contextual", True, 1, "v"),
            (64, 64, 128, 2, True, 0.0, 1.8, "quant", "bias", False, 0, "qk"),
            (32, 128, 64, 4, False, 0.1, 2.2, "euc", "contextual", True, 1, "kv"),
            (128, 128, 256, 8, True, 0.2, 1.2, "product", "contextual", False, 0, "qkv"),
        ],
    )
    def test_forward_pass(
        self,
        in_dim,
        out_dim,
        att_dim,
        n_heads,
        use_mlp,
        dropout,
        rpe_ratio,
        rpe_method,
        rpe_mode,
        rpe_shared_head,
        rpe_skip,
        rpe_on,
    ):
        sample_input = torch.randn(2, 5, in_dim)
        layer = iRPETransformerLayer(
            in_dim,
            out_dim,
            att_dim,
            n_heads,
            use_mlp,
            dropout,
            rpe_ratio,
            rpe_method,
            rpe_mode,
            rpe_shared_head,
            rpe_skip,
            rpe_on,
        )
        output = layer(sample_input)
        expected_out_dim = out_dim if out_dim is not None else in_dim
        assert output.shape == (2, 5, expected_out_dim)

    def test_forward_pass_return_att(self):
        in_dim = 32
        att_dim = 64
        n_heads = 2
        sample_input = torch.randn(2, 5, in_dim)
        layer = iRPETransformerLayer(in_dim, att_dim, n_heads=n_heads)
        output, attention = layer(sample_input, return_att=True)
        assert output.shape == (2, 5, att_dim)
        assert attention.shape == (2, n_heads, 5, 5)  # batch_size, n_heads, bag_size, bag_size


class TestiRPETransformerEncoder:
    @pytest.mark.parametrize(
        "in_dim, out_dim, att_dim, n_heads, n_layers, use_mlp, add_self, dropout, rpe_ratio, rpe_method, rpe_mode, rpe_shared_head, rpe_skip, rpe_on",
        [
            (32, 64, 128, 2, 2, True, False, 0.1, 2.0, "product", "contextual", True, 1, "k"),
            (64, 64, 64, 4, 1, False, True, 0.0, 1.5, "euc", "bias", False, 0, "q"),
            (128, None, 256, 8, 3, True, False, 0.2, 1.0, "cross", "contextual", True, 1, "v"),
            (64, 128, 128, 2, 2, True, False, 0.0, 1.8, "quant", "bias", False, 0, "qk"),
            (32, 32, 64, 4, 1, False, True, 0.1, 2.2, "euc", "contextual", True, 1, "kv"),
            (128, 256, 256, 8, 3, True, False, 0.2, 1.2, "product", "contextual", False, 0, "qkv"),
        ],
    )
    def test_forward_pass(
        self,
        in_dim,
        out_dim,
        att_dim,
        n_heads,
        n_layers,
        use_mlp,
        add_self,
        dropout,
        rpe_ratio,
        rpe_method,
        rpe_mode,
        rpe_shared_head,
        rpe_skip,
        rpe_on,
    ):
        sample_input = torch.randn(2, 5, in_dim)
        encoder = iRPETransformerEncoder(
            in_dim,
            out_dim,
            att_dim,
            n_heads,
            n_layers,
            use_mlp,
            add_self,
            dropout,
            rpe_ratio,
            rpe_method,
            rpe_mode,
            rpe_shared_head,
            rpe_skip,
            rpe_on,
        )
        output = encoder(sample_input)
        expected_out_dim = out_dim if out_dim is not None else in_dim
        assert output.shape == (2, 5, expected_out_dim)

    def test_forward_pass_return_att(self):
        in_dim = 32
        att_dim = 64
        n_heads = 2
        n_layers = 2
        sample_input = torch.randn(2, 5, in_dim)
        encoder = iRPETransformerEncoder(in_dim, att_dim, n_heads, n_layers)
        output, attention = encoder(sample_input, return_att=True)
        assert output.shape == (2, 5, att_dim)
        assert attention[0].shape == (2, n_heads, 5, 5)  # batch_size, n_heads, bag_size, bag_size

    def test_add_self_true(self):
        in_dim = 32
        att_dim = 32
        n_layers = 1
        sample_input = torch.randn(2, 5, in_dim)
        encoder = iRPETransformerEncoder(in_dim, att_dim, n_heads=2, n_layers=n_layers, add_self=True)
        output = encoder(sample_input)
        assert output.shape == (2, 5, in_dim)  # Output should have same shape as input

    def test_encoder_no_layers(self):
        in_dim = 32
        att_dim = 64
        sample_input = torch.randn(2, 5, in_dim)
        encoder = iRPETransformerEncoder(in_dim=in_dim, att_dim=att_dim, n_layers=0, add_self=False)
        output = encoder(sample_input)
        assert output.shape == (2, 5, in_dim)

    def test_rpe_skip_parameter(self):
        in_dim = 32
        att_dim = 64
        n_layers = 2
        rpe_skip = 0
        encoder = iRPETransformerEncoder(in_dim, att_dim, n_heads=2, n_layers=n_layers, rpe_skip=rpe_skip)
        assert encoder.rpe_skip == rpe_skip
        for layer in encoder.layers:
            assert layer.att_module.rpe_skip == rpe_skip

    def test_invalid_rpe_method(self):
        with pytest.raises(ValueError):
            iRPETransformerLayer(in_dim=32, att_dim=64, rpe_method="invalid")

    def test_invalid_rpe_mode(self):
        with pytest.raises(ValueError):
            iRPETransformerLayer(in_dim=32, att_dim=64, rpe_mode="invalid")

    def test_invalid_rpe_skip(self):
        with pytest.raises(ValueError):
            iRPETransformerLayer(in_dim=32, att_dim=64, rpe_skip=2)

    def test_invalid_rpe_on(self):
        with pytest.raises(ValueError):
            iRPETransformerLayer(in_dim=32, att_dim=64, rpe_on="invalid")