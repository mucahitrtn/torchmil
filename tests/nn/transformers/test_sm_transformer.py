import pytest
import torch

from torchmil.nn.transformers.sm_transformer import (
    SmMultiheadSelfAttention,
    SmTransformerEncoder,
    SmTransformerLayer,
)


def test_sm_multihead_self_attention():
    batch_size = 2
    bag_size = 3
    in_dim = 16
    X = torch.randn(batch_size, bag_size, in_dim)
    adj = torch.rand(batch_size, bag_size, bag_size)
    mask = torch.randint(0, 2, (batch_size, bag_size)).bool()
    att_dim = 8
    n_heads = 2

    # Test with default parameters
    att = SmMultiheadSelfAttention(in_dim=in_dim, att_dim=att_dim, n_heads=n_heads)
    output = att(X, adj)
    assert output.shape == (X.shape[0], X.shape[1], in_dim)

    # Test with specified out_dim
    out_dim = 12
    att = SmMultiheadSelfAttention(in_dim=in_dim, out_dim=out_dim, att_dim=att_dim, n_heads=n_heads)
    output = att(X, adj)
    assert output.shape == (X.shape[0], X.shape[1], out_dim)

    # Test with return_att=True
    output, attention_weights = att(X, adj, return_att=True)
    assert output.shape == (X.shape[0], X.shape[1], out_dim)
    assert attention_weights.shape == (X.shape[0], n_heads, X.shape[1], X.shape[1])

    # Test with mask
    output = att(X, adj, mask=mask)
    assert output.shape == (X.shape[0], X.shape[1], out_dim)

    # Test with different sm_alpha and sm_mode
    att = SmMultiheadSelfAttention(in_dim=in_dim, att_dim=att_dim, n_heads=n_heads, sm_alpha=0.5, sm_mode='approx', sm_steps=5)
    output = att(X, adj)
    assert output.shape == (X.shape[0], X.shape[1], in_dim)


def test_sm_transformer_layer():
    batch_size = 2
    bag_size = 3
    in_dim = 16
    X = torch.randn(batch_size, bag_size, in_dim)
    adj = torch.rand(batch_size, bag_size, bag_size)
    mask = torch.randint(0, 2, (batch_size, bag_size)).bool()
    att_dim = 8
    n_heads = 2

    # Test with default parameters
    layer = SmTransformerLayer(in_dim=in_dim, att_dim=att_dim, n_heads=n_heads)
    output = layer(X, adj)
    assert output.shape == (X.shape[0], X.shape[1], in_dim)

    # Test with specified out_dim
    out_dim = 12
    layer = SmTransformerLayer(in_dim=in_dim, out_dim=out_dim, att_dim=att_dim, n_heads=n_heads)
    output = layer(X, adj)
    assert output.shape == (X.shape[0], X.shape[1], out_dim)

    # Test with use_mlp=False
    layer = SmTransformerLayer(in_dim=in_dim, att_dim=att_dim, n_heads=n_heads, use_mlp=False)
    output = layer(X, adj)
    assert output.shape == (X.shape[0], X.shape[1], in_dim)

    # Test with return_att=True
    output, attention_weights = layer(X, adj, return_att=True)
    assert output.shape == (X.shape[0], X.shape[1], in_dim)
    assert attention_weights.shape == (X.shape[0], n_heads, X.shape[1], X.shape[1])

    # Test with mask
    output = layer(X, adj, mask=mask)
    assert output.shape == (X.shape[0], X.shape[1], in_dim)

    # Test with different sm parameters
    layer = SmTransformerLayer(in_dim=in_dim, att_dim=att_dim, n_heads=n_heads, sm_alpha=0.3, sm_mode='exact')
    output = layer(X, adj)
    assert output.shape == (X.shape[0], X.shape[1], in_dim)


def test_sm_transformer_encoder():
    batch_size = 2
    bag_size = 3
    in_dim = 16
    X = torch.randn(batch_size, bag_size, in_dim)
    adj = torch.rand(batch_size, bag_size, bag_size)
    mask = torch.randint(0, 2, (batch_size, bag_size)).bool()
    att_dim = 8
    n_heads = 2
    n_layers = 2

    # Test with default parameters
    encoder = SmTransformerEncoder(in_dim=in_dim, att_dim=att_dim, n_heads=n_heads, n_layers=n_layers)
    output = encoder(X, adj)
    assert output.shape == (X.shape[0], X.shape[1], in_dim)

    # Test with specified out_dim
    out_dim = 12
    encoder = SmTransformerEncoder(in_dim=in_dim, out_dim=out_dim, att_dim=att_dim, n_heads=n_heads, n_layers=n_layers)
    output = encoder(X, adj)
    assert output.shape == (X.shape[0], X.shape[1], out_dim)

    # Test with add_self=True
    encoder = SmTransformerEncoder(in_dim=in_dim, att_dim=in_dim, n_heads=n_heads, n_layers=n_layers, add_self=True)
    output = encoder(X, adj)
    assert output.shape == (X.shape[0], X.shape[1], in_dim)

    # Test with use_mlp=False
    encoder = SmTransformerEncoder(in_dim=in_dim, att_dim=att_dim, n_heads=n_heads, n_layers=n_layers, use_mlp=False)
    output = encoder(X, adj)
    assert output.shape == (X.shape[0], X.shape[1], in_dim)

    # Test with return_att=True
    output, attention_weights = encoder(X, adj, return_att=True)
    assert output.shape == (X.shape[0], X.shape[1], in_dim)
    assert len(attention_weights) == n_layers
    assert attention_weights[0].shape == (X.shape[0], n_heads, X.shape[1], X.shape[1])

    # Test with mask
    output = encoder(X, adj, mask=mask)
    assert output.shape == (X.shape[0], X.shape[1], in_dim)

    # Test with different sm parameters
    encoder = SmTransformerEncoder(in_dim=in_dim, att_dim=att_dim, n_heads=n_heads, n_layers=n_layers, sm_alpha=0.7, sm_mode='approx', sm_steps=8)
    output = encoder(X, adj)
    assert output.shape == (X.shape[0], X.shape[1], in_dim)