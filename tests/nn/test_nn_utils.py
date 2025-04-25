import torch
import pytest

from torchmil.nn.utils import LazyLinear, masked_softmax, MaskedSoftmax, get_feat_dim, SinusoidalPositionalEncodingND, log_sum_exp, delta, SmoothTop1SVM

# LazyLinear tests
def test_lazy_linear_lazy_init():
    linear = LazyLinear(out_features=10)
    x = torch.randn(5, 20)
    result = linear(x)
    assert result.shape == (5, 10)

def test_lazy_linear_eager_init():
    linear = LazyLinear(in_features=20, out_features=10)
    x = torch.randn(5, 20)
    result = linear(x)
    assert result.shape == (5, 10)

# masked_softmax tests
def test_masked_softmax_no_mask():
    x = torch.randn(2, 3, 4)
    result = masked_softmax(x)
    assert result.shape == x.shape

def test_masked_softmax_with_mask():
    x = torch.randn(2, 3, 4)
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    result = masked_softmax(x, mask)
    assert result.shape == x.shape

def test_masked_softmax_mask_dim_mismatch():
    x = torch.randn(2, 3, 4)
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    result = masked_softmax(x, mask)
    assert result.shape == x.shape

# MaskedSoftmax module tests
def test_masked_softmax_module():
    x = torch.randn(2, 3)
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    masked_softmax_module = MaskedSoftmax()
    result = masked_softmax_module(x, mask)
    assert result.shape == x.shape

# get_feat_dim tests
def test_get_feat_dim():
    linear = torch.nn.Linear(10, 20)
    feat_dim = get_feat_dim(linear, (10,))
    assert feat_dim == 20

# SinusoidalPositionalEncodingND tests
def test_sinusoidal_positional_encoding_nd_2d():
    encoding = SinusoidalPositionalEncodingND(n_dim=2, channels=16)
    tensor = torch.randn(2, 4, 4, 16)
    result = encoding(tensor)
    assert result.shape == tensor.shape

def test_sinusoidal_positional_encoding_nd_3d():
    encoding = SinusoidalPositionalEncodingND(n_dim=3, channels=16)
    tensor = torch.randn(2, 4, 4, 4, 16)
    result = encoding(tensor)
    assert result.shape == tensor.shape

def test_sinusoidal_positional_encoding_nd_invalid_shape():
    encoding = SinusoidalPositionalEncodingND(n_dim=2, channels=16)
    tensor = torch.randn(2, 4, 16)
    with pytest.raises(RuntimeError):
        encoding(tensor)

# log_sum_exp tests
def test_log_sum_exp():
    x = torch.randn(3, 5)
    result = log_sum_exp(x)
    assert result.shape == (3,)

# delta tests
def test_delta():
    y = torch.tensor([0, 1, 2])
    y = torch.nn.functional.one_hot(y, num_classes=4)
    labels = torch.tensor([0, 1, 2, 3])
    result = delta(y, labels)
    assert result.shape == (3, 1, 4)

# SmoothTop1SVM tests
def test_smooth_top1_svm():
    loss_fn = SmoothTop1SVM(n_classes=4)
    x = torch.randn(2, 4)
    y = torch.tensor([1, 2])
    # one hot encoding
    y = torch.nn.functional.one_hot(y, num_classes=4)
    result = loss_fn(x, y)
    assert result.shape == ()
