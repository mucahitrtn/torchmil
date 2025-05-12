import torch

class Layer(torch.nn.Module):
    r"""
    Generic Transformer layer class.

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times D}$,
    and (optional) additional arguments, this module computes:

    \begin{align*}
    \mathbf{Z} & = \mathbf{X} + \operatorname{Att}( \operatorname{LayerNorm}(\mathbf{X}) ) \\
    \mathbf{Y} & = \mathbf{Z} + \operatorname{MLP}(\operatorname{LayerNorm}(\mathbf{Z})), \\
    \end{align*}

    and outputs $\mathbf{Y}$.
    $\operatorname{Att}$ is given by the `att_module` argument, and $\operatorname{MLP}$ is given by the `mlp_module` argument.
    """

    def __init__(
        self,
        att_module : torch.nn.Module,
        in_dim: int,
        att_in_dim : int,
        out_dim: int = None,
        att_out_dim : int = None,
        use_mlp: bool = True,
        mlp_module: torch.nn.Module = None,
        dropout: float = 0.0
    ):
        """
        Arguments:
            att_module: Attention module. Assumes input of shape `(batch_size, seq_len, att_in_dim)` and outputs of shape `(batch_size, seq_len, att_out_dim)`.
            in_dim: Input dimension.
            att_in_dim: Input dimension for the attention module.
            out_dim: Output dimension. If None, out_dim = in_dim.
            att_out_dim: Output dimension for the attention module. If None, att_out_dim = in_dim.
            use_mlp: Whether to use a MLP after the attention layer.
            mlp_module: MLP module.
            dropout: Dropout rate.
        """
        super().__init__()

        self.att_module = att_module

        if out_dim is None:
            out_dim = in_dim
        
        if att_out_dim is None:
            att_out_dim = att_in_dim

        if in_dim != att_in_dim:
            self.in_proj = torch.nn.Linear(in_dim, att_in_dim)
        else:
            self.in_proj = torch.nn.Identity()

        if att_in_dim != att_out_dim:
            self.att_proj = torch.nn.Linear(att_in_dim, att_out_dim)
        else:
            self.att_proj = torch.nn.Identity()

        self.use_mlp = use_mlp
        if use_mlp:
            if mlp_module is None:
                self.mlp_module = torch.nn.Sequential(
                    torch.nn.Linear(att_out_dim, 4*att_out_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(4*att_out_dim, att_out_dim),
                    torch.nn.Dropout(dropout)
                )
            else:
                self.mlp_module = mlp_module

        if out_dim != att_out_dim:
            self.out_proj = torch.nn.Linear(att_out_dim, out_dim)
        else:
            self.out_proj = torch.nn.Identity()

        self.norm1 = torch.nn.LayerNorm(att_in_dim)
        self.norm2 = torch.nn.LayerNorm(att_out_dim)

    def forward(
        self,
        X: torch.Tensor,
        return_att: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Arguments:
            X: Input tensor of shape `(batch_size, seq_len, in_dim)`.
            return_att: If True, returns attention weights, of shape `(batch_size, n_heads, seq_len, seq_len)`.
            **kwargs: Additional arguments for the attention module.
        
        Returns:
            Y: Output tensor of shape `(batch_size, seq_len, out_dim)`.
            If `return_att` is True, also returns attention weights, of shape `(batch_size, n_heads, seq_len, seq_len)`.
        """
        X = self.in_proj(X)
        out_att = self.att_module(self.norm1(X), return_att=return_att, **kwargs)
        if return_att:
            Y, att = out_att
        else:
            Y = out_att
        Y = self.att_proj(X) + Y
        if self.use_mlp:
            Y = Y + self.mlp_module(self.norm2(Y))
        Y = self.out_proj(Y)
        if return_att:
            return Y, att
        else:
            return Y
