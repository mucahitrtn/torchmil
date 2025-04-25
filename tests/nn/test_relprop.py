import torch
import pytest
from einops import rearrange

from torchmil.nn.relprop import ( 
    RelPropClone,
    RelPropAdd2,
    RelPropIdentity,
    RelPropLinear,
    RelPropLayerNorm,
    RelPropSequential,
    RelPropReLU,
    RelPropSoftmax,
    RelPropDropout,
    RelPropIndexSelect,
    RelPropEinsum,
    RelPropMultiheadSelfAttention,
    RelPropTransformerLayer,
    RelPropTransformerEncoder,
    safe_divide,
    forward_hook,
    RelPropModule,
    SimpleRelPropModule
)

# Fixtures for test data
@pytest.fixture
def sample_tensor():
    return torch.randn(2, 3, 4)

@pytest.fixture
def sample_input_list_einsum():
    return [torch.randn(2, 3, 4), torch.randn(2, 4, 5)]

@pytest.fixture
def sample_input_list_add2():
    return [torch.randn(2, 3, 4), torch.randn(2, 3, 4)]

@pytest.fixture
def sample_r():
    return torch.randn(2, 3, 4)

@pytest.fixture
def sample_index():
    return torch.tensor([0, 2])

@pytest.fixture
def sample_attention_input():
    return torch.randn(2, 5, 64)  # Example input for attention: batch_size, seq_len, in_dim

# Test RelPropModule and SimpleRelPropModule
def test_relprop_module():
    class TestModule(RelPropModule):
        def _relprop(self, ctx, R, **kwargs):
            return R
    module = TestModule()
    x = torch.randn(2, 3, requires_grad=True)
    out = module(x)
    r = torch.randn(2, 3)
    out_r = module.relprop(r)
    assert torch.allclose(out_r, r)

def test_simple_relprop_module():
    module = SimpleRelPropModule()
    x = torch.randn(2, 3, requires_grad=True)
    out = module(x)
    r = torch.randn(2, 3)
    out_r = module.relprop(r)
    assert torch.allclose(out_r, r)

# Test safe_divide function
def test_safe_divide():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([1.0, 0.0, 2.0, -2.0])
    result = safe_divide(a, b)
    expected = torch.tensor([1.0, 0.0, 1.5, -2.0])
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

# Test RelPropReLU
def test_relprop_relu(sample_tensor):
    relu = RelPropReLU()
    output = relu(sample_tensor)
    R = torch.randn_like(output)
    relprop_output = relu.relprop(R)
    assert relprop_output.shape == sample_tensor.shape

# Test RelPropSoftmax
def test_relprop_softmax(sample_tensor):
    softmax = RelPropSoftmax(dim=-1)
    output = softmax(sample_tensor)
    R = torch.randn_like(output)
    relprop_output = softmax.relprop(R)
    assert relprop_output.shape == sample_tensor.shape

# Test RelPropLayerNorm
def test_relprop_layernorm(sample_tensor):
    norm = RelPropLayerNorm(normalized_shape=sample_tensor.shape[1:])
    output = norm(sample_tensor)
    R = torch.randn_like(output)
    relprop_output = norm.relprop(R)
    assert relprop_output.shape == sample_tensor.shape

# Test RelPropDropout
def test_relprop_dropout(sample_tensor):
    dropout = RelPropDropout(p=0.5)
    output = dropout(sample_tensor)
    R = torch.randn_like(output)
    relprop_output = dropout.relprop(R)
    assert relprop_output.shape == sample_tensor.shape

# Test RelPropIdentity
def test_relprop_identity(sample_tensor):
    identity = RelPropIdentity()
    output = identity(sample_tensor)
    R = torch.randn_like(output)
    relprop_output = identity.relprop(R)
    assert relprop_output.shape == sample_tensor.shape

# Test RelPropSequential
def test_relprop_sequential(sample_tensor):
    seq = RelPropSequential(
        RelPropLinear(sample_tensor.shape[-1], 8),
        RelPropReLU(),
        RelPropLinear(8, sample_tensor.shape[-1])
    )
    output = seq(sample_tensor)
    R = torch.randn_like(output)
    relprop_output = seq.relprop(R)
    assert relprop_output.shape == sample_tensor.shape

# Test RelPropIndexSelect
def test_relprop_index_select(sample_tensor, sample_index):
    index_select = RelPropIndexSelect()
    output = index_select(sample_tensor, dim=1, index=sample_index)
    R = torch.randn_like(output)
    relprop_output = index_select.relprop(R)
    assert relprop_output.shape == sample_tensor.shape

# Test RelPropEinsum
def test_relprop_einsum(sample_input_list_einsum):
    equation = "b i j, b j k -> b i k"
    for a in sample_input_list_einsum:
        print(a.shape)
    einsum = RelPropEinsum(equation)
    output = einsum(*sample_input_list_einsum)
    R = torch.randn_like(output)
    relprop_output = einsum.relprop(R)
    assert isinstance(relprop_output, list)
    assert len(relprop_output) == len(sample_input_list_einsum)
    for i, tensor in enumerate(relprop_output):
        assert tensor.shape == sample_input_list_einsum[i].shape

# Test RelPropClone
def test_relprop_clone(sample_tensor):
    n = 2
    clone = RelPropClone()
    output = clone(sample_tensor, n)
    R = [torch.randn_like(out) for out in output]
    relprop_output = clone.relprop(R)
    assert relprop_output.shape == sample_tensor.shape

# Test RelPropAdd2
def test_relprop_add2(sample_input_list_add2):
    add2 = RelPropAdd2()
    output = add2(*sample_input_list_add2)
    R = torch.randn_like(output)
    relprop_output = add2.relprop(R)
    assert isinstance(relprop_output, tuple)
    assert len(relprop_output) == 2
    assert relprop_output[0].shape == sample_input_list_add2[0].shape
    assert relprop_output[1].shape == sample_input_list_add2[1].shape

# Test RelPropLinear
def test_relprop_linear(sample_tensor):
    in_features = sample_tensor.shape[-1]
    out_features = 8
    linear = RelPropLinear(in_features, out_features)
    output = linear(sample_tensor)
    R = torch.randn_like(output)
    relprop_output = linear.relprop(R)
    assert relprop_output.shape == sample_tensor.shape

# Test RelPropMultiheadSelfAttention
def test_relprop_multihead_self_attention(sample_attention_input):
    in_dim = sample_attention_input.shape[-1]
    att_dim = 64
    n_heads = 4
    dropout = 0.1
    attention = RelPropMultiheadSelfAttention(in_dim, att_dim=att_dim, n_heads=n_heads, dropout=dropout)
    output = attention(sample_attention_input)
    R = torch.randn_like(output)
    relprop_output = attention.relprop(R)
    assert relprop_output.shape == sample_attention_input.shape

def test_relprop_multihead_self_attention_return_attention_relevance(sample_attention_input):
    in_dim = sample_attention_input.shape[-1]
    att_dim = 64
    n_heads = 4
    dropout = 0.1
    attention = RelPropMultiheadSelfAttention(in_dim, att_dim=att_dim, n_heads=n_heads, dropout=dropout)
    output = attention(sample_attention_input)
    R = torch.randn_like(output)
    relprop_output, attention_relevance = attention.relprop(R, return_att_relevance=True)
    assert relprop_output.shape == sample_attention_input.shape
    assert attention_relevance.shape == (
        2,
        n_heads,
        sample_attention_input.shape[1],
        sample_attention_input.shape[1],
    )

# Test RelPropTransformerLayer
def test_relprop_transformer_layer(sample_attention_input):
    in_dim = sample_attention_input.shape[-1]
    att_dim = 64
    n_heads = 4
    use_mlp = True
    dropout = 0.1
    layer = RelPropTransformerLayer(in_dim, att_dim=att_dim, n_heads=n_heads, use_mlp=use_mlp, dropout=dropout)
    output = layer(sample_attention_input)
    R = torch.randn_like(output)
    relprop_output = layer.relprop(R)
    assert relprop_output.shape == sample_attention_input.shape

def test_relprop_transformer_layer_return_attention_relevance(sample_attention_input):
    in_dim = sample_attention_input.shape[-1]
    att_dim = 64
    n_heads = 4
    use_mlp = True
    dropout = 0.1
    layer = RelPropTransformerLayer(in_dim, att_dim=att_dim, n_heads=n_heads, use_mlp=use_mlp, dropout=dropout)
    output = layer(sample_attention_input)
    R = torch.randn_like(output)
    relprop_output, attention_relevance = layer.relprop(R, return_att_relevance=True)
    assert relprop_output.shape == sample_attention_input.shape
    assert attention_relevance.shape == (
        2,
        n_heads,
        sample_attention_input.shape[1],
        sample_attention_input.shape[1],
    )

# Test RelPropTransformerEncoder
def test_relprop_transformer_encoder(sample_attention_input):
    in_dim = sample_attention_input.shape[-1]
    att_dim = 64
    n_heads = 4
    n_layers = 2
    use_mlp = True
    dropout = 0.1
    encoder = RelPropTransformerEncoder(in_dim, att_dim=att_dim, n_heads=n_heads, n_layers=n_layers, use_mlp=use_mlp, dropout=dropout)
    output = encoder(sample_attention_input)
    R = torch.randn_like(output)
    relprop_output = encoder.relprop(R)
    assert relprop_output.shape == sample_attention_input.shape

def test_relprop_transformer_encoder_return_attention_relevance(sample_attention_input):
    in_dim = sample_attention_input.shape[-1]
    att_dim = 64
    n_heads = 4
    n_layers = 2
    use_mlp = True
    dropout = 0.1
    encoder = RelPropTransformerEncoder(in_dim, att_dim=att_dim, n_heads=n_heads, n_layers=n_layers, use_mlp=use_mlp, dropout=dropout)
    output = encoder(sample_attention_input)
    R = torch.randn_like(output)
    relprop_output, attention_relevance_list = encoder.relprop(R, return_att_relevance=True)
    assert relprop_output.shape == sample_attention_input.shape
    assert len(attention_relevance_list) == n_layers
    for attention_relevance in attention_relevance_list:
        assert attention_relevance.shape == (
            2,
            n_heads,
            sample_attention_input.shape[1],
            sample_attention_input.shape[1],
        )
