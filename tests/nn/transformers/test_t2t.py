import torch
import pytest

from torchmil.nn.transformers import T2TLayer 

@pytest.mark.parametrize(
    "batch_size, seq_len, in_dim, att_dim, kernel_size, stride, padding, dilation, n_heads, use_mlp",
    [
        (2, 16, 32, 64, (3, 3), (1, 1), (1, 1), (1, 1), 4, True),
        (4, 25, 64, 128, (2, 2), (2, 2), (0, 0), (1, 1), 8, False),
        (1, 49, 128, 256, (1, 1), (1, 1), (0, 0), (1, 1), 2, True),
        (3, 100, 256, 512, (3, 3), (1, 1), (2, 2), (1, 1), 1, False),
    ],
)
def test_t2t_layer_forward(
    batch_size,
    seq_len,
    in_dim,
    att_dim,
    kernel_size,
    stride,
    padding,
    dilation,
    n_heads,
    use_mlp,
):
    """Tests the forward pass of T2TLayer."""
    layer = T2TLayer(
        in_dim=in_dim,
        att_dim=att_dim,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        n_heads=n_heads,
        use_mlp=use_mlp,
    )
    x = torch.randn(batch_size, seq_len, in_dim)
    output = layer(x)

    # Calculate expected output sequence length
    s = int(seq_len**0.5)  # Original feature map size (assuming square input)
    new_s = (s + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    expected_out_dim = att_dim * kernel_size[0] * kernel_size[1]
    expected_output_shape = (batch_size, new_s * new_s, expected_out_dim)
    assert output.shape == expected_output_shape

@pytest.mark.parametrize(
    "batch_size, seq_len, in_dim, att_dim, kernel_size, stride, padding, dilation, n_heads, use_mlp, out_dim",
    [
        (2, 16, 32, 64, (3, 3), (1, 1), (1, 1), (1, 1), 4, True, 128),
        (4, 25, 64, 128, (2, 2), (2, 2), (0, 0), (1, 1), 8, False, 256),
        (1, 49, 128, 256, (1, 1), (1, 1), (0, 0), (1, 1), 2, True, 512),
        (3, 100, 256, 512, (3, 3), (1, 1), (2, 2), (1, 1), 1, False, 1024),
    ],
)
def test_t2t_layer_forward_with_out_dim(
    batch_size,
    seq_len,
    in_dim,
    att_dim,
    kernel_size,
    stride,
    padding,
    dilation,
    n_heads,
    use_mlp,
    out_dim,
):
    """Tests the forward pass of T2TLayer with a specified out_dim."""
    layer = T2TLayer(
        in_dim=in_dim,
        att_dim=att_dim,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        n_heads=n_heads,
        use_mlp=use_mlp,
        out_dim=out_dim,
    )
    x = torch.randn(batch_size, seq_len, in_dim)
    output = layer(x)

    # Calculate expected output sequence length
    s = int(seq_len**0.5)  # Original feature map size (assuming square input)
    new_s = (s + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    expected_output_shape = (batch_size, new_s * new_s, out_dim)
    assert output.shape == expected_output_shape
