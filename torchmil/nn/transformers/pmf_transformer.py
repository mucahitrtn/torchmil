import torch

from .conventional_transformer import TransformerLayer, Encoder

class PMFTransformerLayer(torch.nn.Module):
    r"""
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
            out_dim: Output dimension.
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
        transf_in_dim = in_dim
        for d in kernel_size:
            transf_in_dim *= d
        self.transf_in_dim = transf_in_dim
        self.transf_layer = TransformerLayer(in_dim=transf_in_dim, att_dim=att_dim, out_dim=out_dim, n_heads=n_heads, use_mlp=use_mlp, dropout=dropout)
    
    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        Arguments:
            X: Input tensor of shape `(batch_size, in_dim, ...)`.
        Returns:
            Y: Output tensor of shape `(batch_size, embed_dim, ...)`.
        """

        assert len(X.shape) <= 4, "Input tensor must have at most 4 dimensions. Only batched sequence-like or image-like tensors are supported."

        if len(X.shape) == 3:
            X = X.unsqueeze(-1) # (batch_size, in_dim, L, 1)
        
        X = self.unfold(X) # (batch_size, in_dim * kernel_size[0] * kernel_size[1], L)
        X = X.transpose(1, 2) # (batch_size, L, in_dim * kernel_size[0] * kernel_size[1])

        X = self.transf_layer(X) # (batch_size, L, embed_dim)

        return X

class PMFTransformerEncoder(torch.nn.Module):
    r"""
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        att_dim: int = 512,
        kernel_list: list[tuple[int, int]] = [(3, 3), (5, 5), (7, 7)],
        stride_list: list[tuple[int, int]] = [(1, 1), (1, 1), (1, 1)],
        padding_list: list[tuple[int, int]] = [(1, 1), (2, 2), (3, 3)],
        dilation_list: list[tuple[int, int]] = [(1, 1), (1, 1), (1, 1)],
        n_heads: int = 4,
        use_mlp: bool = True,
        dropout: float = 0.0
    ):
        """
        Arguments:
            in_dim: Input dimension.
            out_dim: Output dimension.
            att_dim: Attention dimension.
            kernel_list: List of kernel sizes.
            stride_list: List of strides.
            padding_list: List of paddings.
            dilation_list: List of dilations.
            n_heads: Number of heads.
            use_mlp: Whether to use feedforward layer.
            dropout: Dropout rate.
        """

        super().__init__()

        self.layers = torch.nn.ModuleList([
            PMFTransformerLayer(
                in_dim=in_dim if i == 0 else att_dim, 
                att_dim=att_dim, 
                out_dim=out_dim,
                kernel_size=kernel_list[i], stride=stride_list[i], padding=padding_list[i], dilation=dilation_list[i],
                n_heads=n_heads, use_mlp=use_mlp, dropout=dropout
            )
            for i in range(len(kernel_list))
        ])
    
    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        Arguments:
            X: Input tensor of shape `(batch_size, in_dim, coord1, coord2, ..., coordN)`.
        Returns:
            Y: Output tensor of shape `(batch_size, out_dim, L)`.
        """
        X_ = []
        for layer in self.layers:
            X_.append(layer(X)) # (batch_size, L, out_dim)
        X_ = torch.cat(X_, dim=1) # (batch_size, L', out_dim)
        return X_
