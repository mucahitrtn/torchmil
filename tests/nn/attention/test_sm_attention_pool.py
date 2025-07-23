import pytest
import torch

from torchmil.nn.attention.sm_attention_pool import SmAttentionPool
from torchmil.nn.sm import Sm


@pytest.fixture
def sample_inputs_sm_attention_pool():
    """Provides sample input tensors for SmAttentionPool testing."""
    batch_size = 2
    bag_size = 5
    in_dim = 128
    att_dim = 64

    X = torch.randn(batch_size, bag_size, in_dim)
    adj = torch.randint(0, 2, (batch_size, bag_size, bag_size)).float()
    mask = torch.randint(0, 2, (batch_size, bag_size)).bool()

    return X, adj, mask, in_dim, att_dim, batch_size, bag_size


def test_sm_attention_pool_initialization_default(sample_inputs_sm_attention_pool):
    """Test SmAttentionPool initialization with default parameters."""
    _, _, _, in_dim, att_dim, _, _ = sample_inputs_sm_attention_pool
    model = SmAttentionPool(in_dim=in_dim)

    assert isinstance(model, SmAttentionPool)
    assert isinstance(model.sm, Sm)
    assert isinstance(model.proj1, torch.nn.Linear)
    assert isinstance(model.proj2, torch.nn.Linear)


def test_sm_attention_pool_initialization_custom(sample_inputs_sm_attention_pool):
    """Test SmAttentionPool initialization with custom parameters."""
    _, _, _, in_dim, _, _, _ = sample_inputs_sm_attention_pool

    att_dim = 32
    act = "tanh"
    sm_mode = "exact"
    sm_alpha = 0.5
    sm_steps = 5
    sm_where = "late"
    spectral_norm = True

    model = SmAttentionPool(
        in_dim=in_dim,
        att_dim=att_dim,
        act=act,
        sm_mode=sm_mode,
        sm_alpha=sm_alpha,
        sm_steps=sm_steps,
        sm_where=sm_where,
        spectral_norm=spectral_norm,
    )

    assert isinstance(model, SmAttentionPool)
    assert isinstance(model.sm, Sm)
    assert isinstance(model.proj1, torch.nn.Linear)
    assert isinstance(model.proj2, torch.nn.Linear)


def test_sm_attention_pool_initialization_invalid_act(sample_inputs_sm_attention_pool):
    """Test SmAttentionPool initialization with an invalid activation function."""
    _, _, _, in_dim, _, _, _ = sample_inputs_sm_attention_pool
    with pytest.raises(ValueError, match="act must be 'tanh', 'relu' or 'gelu'"):
        SmAttentionPool(in_dim=in_dim, act="invalid_act")


@pytest.mark.parametrize("sm_where", ["early", "mid", "late"])
def test_sm_attention_pool_forward_no_mask_return_att_false(
    sample_inputs_sm_attention_pool, sm_where
):
    """Test forward pass without mask and return_att=False for different sm_where modes."""
    X, adj, _, in_dim, _, batch_size, _ = sample_inputs_sm_attention_pool
    model = SmAttentionPool(in_dim=in_dim, sm_where=sm_where)

    # Ensure model is in eval mode for consistent behavior, though not strictly necessary for shape checks
    model.eval()
    with torch.no_grad():
        z = model(X, adj, mask=None, return_att=False)

    assert isinstance(z, torch.Tensor)
    assert z.shape == (batch_size, in_dim)


@pytest.mark.parametrize("sm_where", ["early", "mid", "late"])
def test_sm_attention_pool_forward_with_mask_return_att_false(
    sample_inputs_sm_attention_pool, sm_where
):
    """Test forward pass with mask and return_att=False for different sm_where modes."""
    X, adj, mask, in_dim, _, batch_size, _ = sample_inputs_sm_attention_pool
    model = SmAttentionPool(in_dim=in_dim, sm_where=sm_where)

    model.eval()
    with torch.no_grad():
        z = model(X, adj, mask=mask, return_att=False)

    assert isinstance(z, torch.Tensor)
    assert z.shape == (batch_size, in_dim)


@pytest.mark.parametrize("sm_where", ["early", "mid", "late"])
def test_sm_attention_pool_forward_no_mask_return_att_true(
    sample_inputs_sm_attention_pool, sm_where
):
    """Test forward pass without mask and return_att=True for different sm_where modes."""
    X, adj, _, in_dim, _, batch_size, bag_size = sample_inputs_sm_attention_pool
    model = SmAttentionPool(in_dim=in_dim, sm_where=sm_where)

    model.eval()
    with torch.no_grad():
        z, f = model(X, adj, mask=None, return_att=True)

    assert isinstance(z, torch.Tensor)
    assert z.shape == (batch_size, in_dim)
    assert isinstance(f, torch.Tensor)
    assert f.shape == (batch_size, bag_size)


@pytest.mark.parametrize("sm_where", ["early", "mid", "late"])
def test_sm_attention_pool_forward_with_mask_return_att_true(
    sample_inputs_sm_attention_pool, sm_where
):
    """Test forward pass with mask and return_att=True for different sm_where modes."""
    X, adj, mask, in_dim, _, batch_size, bag_size = sample_inputs_sm_attention_pool
    model = SmAttentionPool(in_dim=in_dim, sm_where=sm_where)

    model.eval()
    with torch.no_grad():
        z, f = model(X, adj, mask=mask, return_att=True)

    assert isinstance(z, torch.Tensor)
    assert z.shape == (batch_size, in_dim)
    assert isinstance(f, torch.Tensor)
    assert f.shape == (batch_size, bag_size)
