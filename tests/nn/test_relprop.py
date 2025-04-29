import pytest
import torch
import torch.nn as nn

from torchmil.nn.relprop import (
    safe_divide,
    forward_hook,
    RelProp,
    RelPropSimple,
    AddEye,
    ReLU,
    GELU,
    Softmax,
    LayerNorm,
    Dropout,
    MaxPool2d,
    AdaptiveAvgPool2d,
    AvgPool2d,
    Add,
    Identity,
    Einsum,
    IndexSelect,
    Clone,
    Cat,
    Sequential,
    BatchNorm2d,
    Linear,
    MultiheadSelfAttention,
    TransformerLayer,
    TransformerEncoder,
)

# --- Helper functions for testing ---
def assert_tensor_equal(t1, t2):
    assert torch.allclose(t1, t2), f"Tensor mismatch: \n{t1}\nvs\n{t2}"

# --- Tests for individual components ---

def test_safe_divide():
    a = torch.tensor([2.0, 4.0, 6.0])
    b = torch.tensor([1.0, 2.0, 0.0])
    expected = torch.tensor([2.0, 2.0, 0.0])
    assert_tensor_equal(safe_divide(a, b), expected)

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    expected = torch.tensor([[0.0, 2.0], [1.5, 0.0]])
    assert_tensor_equal(safe_divide(a, b), expected)

def test_forward_hook():
    linear = nn.Linear(5, 3)
    hook = linear.register_forward_hook(forward_hook)
    input_tensor = torch.randn(2, 5)
    output_tensor = linear(input_tensor)
    assert hasattr(linear, 'X')
    assert_tensor_equal(linear.X, input_tensor.detach().requires_grad_(True))
    assert hasattr(linear, 'Y')
    assert_tensor_equal(linear.Y, output_tensor)
    hook.remove()

def test_relu_relprop():
    relu = ReLU()
    input_tensor = torch.randn(2, 3, requires_grad=True)
    output_tensor = relu(input_tensor)
    relevance = torch.rand_like(output_tensor)
    propagated_relevance = relu.relprop(relevance)
    assert_tensor_equal(propagated_relevance, relevance) # Simple relprop for ReLU

def test_gelu_relprop():
    gelu = GELU()
    input_tensor = torch.randn(2, 3, requires_grad=True)
    output_tensor = gelu(input_tensor)
    relevance = torch.rand_like(output_tensor)
    propagated_relevance = gelu.relprop(relevance)
    assert_tensor_equal(propagated_relevance, relevance) # Simple relprop for GELU

def test_softmax_relprop():
    softmax = Softmax(dim=1)
    input_tensor = torch.randn(2, 3, requires_grad=True)
    output_tensor = softmax(input_tensor)
    relevance = torch.rand_like(output_tensor)
    propagated_relevance = softmax.relprop(relevance)
    assert_tensor_equal(propagated_relevance, relevance) # Simple relprop for Softmax

def test_dropout_relprop():
    dropout = Dropout(p=0.5)
    input_tensor = torch.randn(2, 3, requires_grad=True)
    output_tensor = dropout(input_tensor)
    relevance = torch.rand_like(output_tensor)
    propagated_relevance = dropout.relprop(relevance)
    assert_tensor_equal(propagated_relevance, relevance) # Simple relprop for Dropout

def test_layernorm_relprop():
    layernorm = LayerNorm(3)
    input_tensor = torch.randn(2, 3, requires_grad=True)
    output_tensor = layernorm(input_tensor)
    relevance = torch.rand_like(output_tensor)
    propagated_relevance = layernorm.relprop(relevance)
    assert_tensor_equal(propagated_relevance, relevance) # Simple relprop for LayerNorm

def test_maxpool2d_relprop():
    maxpool = MaxPool2d(kernel_size=2)
    input_tensor = torch.randn(1, 1, 4, 4, requires_grad=True)
    output_tensor = maxpool(input_tensor)
    relevance = torch.rand_like(output_tensor)
    propagated_relevance = maxpool.relprop(relevance)
    assert propagated_relevance.shape == input_tensor.shape

def test_adaptiveavgpool2d_relprop():
    pool = AdaptiveAvgPool2d((2, 2))
    input_tensor = torch.randn(1, 1, 4, 4, requires_grad=True)
    output_tensor = pool(input_tensor)
    relevance = torch.rand_like(output_tensor)
    propagated_relevance = pool.relprop(relevance)
    assert propagated_relevance.shape == input_tensor.shape

def test_avgpool2d_relprop():
    avgpool = AvgPool2d(kernel_size=2)
    input_tensor = torch.randn(1, 1, 4, 4, requires_grad=True)
    output_tensor = avgpool(input_tensor)
    relevance = torch.rand_like(output_tensor)
    propagated_relevance = avgpool.relprop(relevance)
    assert propagated_relevance.shape == input_tensor.shape

def test_add_relprop():
    add_module = Add()
    input1 = torch.randn(2, 3, requires_grad=True)
    input2 = torch.randn(2, 3, requires_grad=True)
    output_tensor = add_module([input1, input2])
    relevance = torch.rand_like(output_tensor)
    propagated_relevances = add_module.relprop(relevance)
    assert isinstance(propagated_relevances, list)
    assert len(propagated_relevances) == 2
    assert propagated_relevances[0].shape == input1.shape
    assert propagated_relevances[1].shape == input2.shape

def test_identity_relprop():
    identity = Identity()
    input_tensor = torch.randn(2, 3, requires_grad=True)
    output_tensor = identity(input_tensor)
    relevance = torch.rand_like(output_tensor)
    propagated_relevance = identity.relprop(relevance)
    assert_tensor_equal(propagated_relevance, relevance)

def test_einsum_relprop():
    einsum_module = Einsum("b i j, b j k -> b i k")
    input1 = torch.randn(2, 3, 4, requires_grad=True)
    input2 = torch.randn(2, 4, 5, requires_grad=True)
    output_tensor = einsum_module([input1, input2])
    relevance = torch.rand_like(output_tensor)
    propagated_relevances = einsum_module.relprop(relevance)
    assert isinstance(propagated_relevances, list)
    assert len(propagated_relevances) == 2
    assert propagated_relevances[0].shape == input1.shape
    assert propagated_relevances[1].shape == input2.shape

def test_indexselect_relprop():
    index_select = IndexSelect()
    input_tensor = torch.randn(2, 5, requires_grad=True)
    dim = 1
    indices = torch.tensor([0, 2, 4])
    output_tensor = index_select(input_tensor, dim, indices)
    relevance = torch.rand_like(output_tensor)
    propagated_relevance = index_select.relprop(relevance)
    assert propagated_relevance.shape == input_tensor.shape

def test_clone_relprop():
    clone_module = Clone()
    input_tensor = torch.randn(2, 3, requires_grad=True)
    num_clones = 2
    output_tensors = clone_module(input_tensor, num_clones)
    relevances = [torch.randn_like(t) for t in output_tensors]
    propagated_relevance = clone_module.relprop(relevances)
    assert propagated_relevance.shape == input_tensor.shape

def test_cat_relprop():
    cat_module = Cat()
    input1 = torch.randn(2, 3, requires_grad=True)
    input2 = torch.randn(2, 4, requires_grad=True)
    dim = 1
    output_tensor = cat_module([input1, input2], dim)
    relevance = torch.rand_like(output_tensor)
    propagated_relevances = cat_module.relprop(relevance)
    assert isinstance(propagated_relevances, list)
    assert len(propagated_relevances) == 2
    assert propagated_relevances[0].shape == input1.shape
    assert propagated_relevances[1].shape == input2.shape

def test_sequential_relprop():
    seq_module = Sequential(
        Linear(5, 3),
        ReLU(),
        Linear(3, 2)
    )
    input_tensor = torch.randn(2, 5, requires_grad=True)
    output_tensor = seq_module(input_tensor)
    relevance = torch.rand_like(output_tensor)
    propagated_relevance = seq_module.relprop(relevance)
    assert propagated_relevance.shape == input_tensor.shape

def test_batchnorm2d_relprop():
    batchnorm = BatchNorm2d(3)
    input_tensor = torch.randn(2, 3, 4, 4, requires_grad=True)
    output_tensor = batchnorm(input_tensor)
    relevance = torch.rand_like(output_tensor)
    propagated_relevance = batchnorm.relprop(relevance)
    assert propagated_relevance.shape == input_tensor.shape

def test_linear_relprop():
    linear = Linear(5, 3)
    input_tensor = torch.randn(2, 5, requires_grad=True)
    output_tensor = linear(input_tensor)
    relevance = torch.rand_like(output_tensor)
    propagated_relevance = linear.relprop(relevance)
    assert propagated_relevance.shape == input_tensor.shape

def test_add_eye_forward():
    add_eye = AddEye()
    input_tensor = torch.randn(2, 3, 4, 4)
    output_tensor = add_eye(input_tensor)
    expected = input_tensor + torch.eye(4).expand_as(input_tensor).to(input_tensor.device)
    assert_tensor_equal(output_tensor, expected)

def test_relprop_simple():
    class DummyModule(RelPropSimple):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 3)

        def forward(self, x):
            return self.linear(x)

    dummy = DummyModule()
    input_tensor = torch.randn(2, 5, requires_grad=True)
    output_tensor = dummy(input_tensor)
    relevance = torch.rand_like(output_tensor)
    propagated_relevance = dummy.relprop(relevance)
    assert propagated_relevance.shape == input_tensor.shape

def test_relprop_base():
    class DummyModule(RelProp):
        def forward(self, x):
            return x * 2

    dummy = DummyModule()
    input_tensor = torch.randn(2, 3, requires_grad=True)
    output_tensor = dummy(input_tensor)
    relevance = torch.rand_like(output_tensor)
    propagated_relevance = dummy.relprop(relevance)
    assert_tensor_equal(propagated_relevance, relevance) # Base relprop returns R

def test_multihead_self_attention_forward():
    msa = MultiheadSelfAttention(in_dim=10, att_dim=8, out_dim=8, n_heads=2)
    input_tensor = torch.randn(2, 5, 10)
    output_tensor = msa(input_tensor)
    assert output_tensor.shape == (2, 5, 8)

def test_multihead_self_attention_relprop():
    msa = MultiheadSelfAttention(in_dim=10, att_dim=8, n_heads=2)
    input_tensor = torch.randn(2, 5, 10, requires_grad=True)
    output_tensor = msa(input_tensor)
    relevance = torch.randn_like(output_tensor)
    propagated_relevance = msa.relprop(relevance)
    assert propagated_relevance.shape == input_tensor.shape

def test_transformer_layer_forward():
    transformer_layer = TransformerLayer(in_dim=10, att_dim=8, out_dim=8, n_heads=2)
    input_tensor = torch.randn(2, 5, 10)
    output_tensor = transformer_layer(input_tensor)
    assert output_tensor.shape == (2, 5, 8)

def test_transformer_layer_relprop():
    transformer_layer = TransformerLayer(in_dim=10, att_dim=8, out_dim=8, n_heads=2)
    input_tensor = torch.randn(2, 5, 10, requires_grad=True)
    output_tensor = transformer_layer(input_tensor)
    relevance = torch.randn_like(output_tensor)
    propagated_relevance = transformer_layer.relprop(relevance)
    assert propagated_relevance.shape == input_tensor.shape

def test_transformer_encoder_forward():
    transformer_encoder = TransformerEncoder(in_dim=10, att_dim=8, out_dim=8, n_heads=2, n_layers=2)
    input_tensor = torch.randn(2, 5, 10)
    output_tensor = transformer_encoder(input_tensor)
    assert output_tensor.shape == (2, 5, 8)

def test_transformer_encoder_relprop():
    transformer_encoder = TransformerEncoder(in_dim=10, att_dim=8, n_heads=2, n_layers=2)
    input_tensor = torch.randn(2, 5, 10, requires_grad=True)
    output_tensor = transformer_encoder(input_tensor)
    relevance = torch.randn_like(output_tensor)
    propagated_relevance = transformer_encoder.relprop(relevance)
    assert propagated_relevance.shape == input_tensor.shape