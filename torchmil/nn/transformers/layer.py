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
        in_dim: int,
        att_dim : int,
        att_module : torch.nn.Module,
        out_dim: int = None,
        use_mlp: bool = True,
        mlp_module: torch.nn.Module = None,
        dropout: float = 0.0
    ):
        """
        Arguments:
            in_dim: Input dimension.
            att_dim: Attention dimension.
            att_module: Attention module.
            out_dim: Output dimension. If None, out_dim = in_dim.
            use_mlp: Whether to use a MLP after the attention layer.
            mlp_module: MLP module.
            dropout: Dropout rate.
        """
        super().__init__()

        self.att_module = att_module

        if out_dim is None:
            out_dim = in_dim

        if in_dim != att_dim:
            self.in_proj = torch.nn.Linear(in_dim, att_dim)
        else:
            self.in_proj = torch.nn.Identity()

        self.use_mlp = use_mlp
        if use_mlp:
            if mlp_module is None:
                self.mlp_module = torch.nn.Sequential(
                    torch.nn.Linear(att_dim, 4*att_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(4*att_dim, att_dim),
                    torch.nn.Dropout(dropout)
                )
            else:
                self.mlp_module = mlp_module

        if out_dim != att_dim:
            self.out_proj = torch.nn.Linear(att_dim, out_dim)
        else:
            self.out_proj = torch.nn.Identity()

        self.norm1 = torch.nn.LayerNorm(in_dim)
        self.norm2 = torch.nn.LayerNorm(att_dim)

    def forward(
        self,
        X: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Arguments:
            X: Input tensor of shape `(batch_size, seq_len, in_dim)`.
        """
        out_att = self.att_module(self.norm1(X), **kwargs)
        if isinstance(out_att, tuple):
            Y = self.in_proj(X) + out_att[0]
        else:
            Y = self.in_proj(X) + out_att
        if self.use_mlp:
            Y = Y + self.mlp_module(self.norm2(Y))
        Y = self.out_proj(Y)
        if isinstance(out_att, tuple):
            return (Y,) + out_att[1:]
        else:
            return Y
