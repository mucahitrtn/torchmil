import pytest
import torch

from torchmil.nn.attention.attention_pool import AttentionPool


class TestAttentionPool:
    @pytest.mark.parametrize("in_dim, att_dim, act, gated",
                             [(10, 128, 'tanh', False),
                              (15, 64, 'relu', True),
                              (8, 256, 'gelu', False)])
    def test_forward_pass(self, in_dim, att_dim, act, gated):
        sample_input = torch.randn(2, 5, in_dim)  # batch_size, bag_size, in_dim
        pool = AttentionPool(in_dim, att_dim, act, gated)
        output = pool(sample_input)
        assert output.shape == (2, in_dim)

    def test_forward_pass_with_mask(self):
        in_dim = 10
        sample_input = torch.randn(2, 5, in_dim)  # batch_size, bag_size, in_dim
        sample_mask = torch.randint(0, 2, (2, 5)).bool()  # batch_size, bag_size
        pool = AttentionPool(in_dim)
        output = pool(sample_input, mask=sample_mask)
        assert output.shape == (2, in_dim)

    def test_forward_pass_return_att(self):
        in_dim = 10
        sample_input = torch.randn(2, 5, in_dim)  # batch_size, bag_size, in_dim
        pool = AttentionPool(in_dim)
        z, attention = pool(sample_input, return_att=True)
        assert z.shape == (2, in_dim)
        assert attention.shape == (2, 5)

    def test_invalid_act_value(self):
        with pytest.raises(ValueError):
            AttentionPool(in_dim=10, act='invalid')

    def test_lazy_initialization(self):
        sample_input = torch.randn(2, 5, 10)
        pool = AttentionPool()  # in_dim is None
        output = pool(sample_input)  # shape: (2, 5, 10)
        assert output.shape == (2, 10) # infers in_dim from the input

    def test_mask_all_zeros(self):
        in_dim = 10
        sample_input = torch.randn(2, 5, in_dim)
        mask = torch.zeros((2, 5), dtype=torch.bool)
        pool = AttentionPool(in_dim)
        output = pool(sample_input, mask=mask)
        assert output.shape == (2, in_dim)

    def test_large_input_values(self):
        in_dim = 10
        large_input = torch.randn(2, 5, in_dim) * 1000  # Create very large input values
        pool = AttentionPool(in_dim)
        output = pool(large_input)
        assert output.shape == (2, in_dim)
