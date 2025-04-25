import torch

from .conventional_transformer import TransformerLayer

class T2TLayer(torch.nn.Module):
    r"""
    Tokens-to-Token (T2T) Transformer layer from [Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](https://arxiv.org/abs/2101.11986)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        att_dim: int = 512,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (2, 2),
        dilation: tuple[int, int] = (1, 1),
        n_heads: int = 4,
        use_mlp: bool = True,
        dropout: float = 0.0
    ):
        """
        Arguments:
            in_dim: Input dimension.
            out_dim: Output dimension. If None, output dimension will be `kernel_size[0] * kernel_size[1] * att_dim`.
            att_dim: Attention dimension.
            kernel_size: Kernel size.
            stride: Stride.
            padding: Padding.
            dilation: Dilation.
            n_heads: Number of heads.
            use_mlp: Whether to use feedforward layer.
            dropout: Dropout rate.
        """
        super().__init__()

        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        self.transf_layer = TransformerLayer(
            in_dim=in_dim,
            att_dim=att_dim,
            out_dim=att_dim,
            n_heads=n_heads,
            use_mlp=use_mlp,
            dropout=dropout
        )

        if out_dim is not None:
            self.out_proj = torch.nn.Linear(att_dim*kernel_size[0]*kernel_size[1], out_dim)
        else:
            self.out_proj = torch.nn.Identity()

    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        Arguments:
            X: Input tensor of shape `(batch_size, seq_len, in_dim)`.
        Returns:
            Y: Output tensor of shape `(batch_size, new_seq_len, out_dim)`. If `out_dim` is None, `out_dim` will be `att_dim * kernel_size[0] * kernel_size[1]`.
        """

        # transformer layer
        X = self.transf_layer(X) # (batch_size, seq_len, att_dim)

        # reshape
        s = int(X.size(1) ** 0.5)
        X = X.transpose(1, 2) # (batch_size, att_dim, seq_len)
        X = X.reshape(X.size(0), X.size(1), s, s) # (batch_size, att_dim, s, s)

        # unfold
        X = self.unfold(X) # (batch_size, att_dim * kernel_size[0] * kernel_size[1], new_seq_len)
        X = X.transpose(1, 2) # (batch_size, new_seq_len, att_dim * kernel_size[0] * kernel_size[1])
        X = self.out_proj(X) # (batch_size, new_seq_len, out_dim)

        return X
