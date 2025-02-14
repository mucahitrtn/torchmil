import torch

from .layers import TransformerLayer, SmTransformerLayer, NystromTransformerLayer


class Encoder(torch.nn.Module):
    """
    Generic encoder class.    
    """

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        add_self: bool = False
    ):
        """
        Class constructor

        Arguments:
            layers: List of encoder layers.
            add_self: Whether to add input to output.
        """
        super(Encoder, self).__init__()
        self.add_self = add_self
        self.layers = layers

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward method.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.
            mask: Mask tensor of shape `(batch_size, bag_size)`.
            **kwargs: Additional arguments.

        Returns:
            Y: Output tensor of shape `(batch_size, bag_size, in_dim)`.        
        """

        Y = X  # (batch_size, bag_size, in_dim)
        for layer in self.layers:
            Y = layer(Y, mask=mask, **kwargs)
            if self.add_self:
                Y = Y + X
        return Y


class TransformerEncoder(Encoder):
    r"""
    A Transformer encoder with skip connections and layer normalization.

    Given an input bag input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times D}$,
    it computes:

    \begin{align*}
    \mathbf{X}^{0} & = \mathbf{X} \\
    \mathbf{Z}^{l} & = \mathbf{X}^{l-1} + \text{SelfAttention}( \text{LayerNorm}(\mathbf{X}^{l-1}) ), \quad l = 1, \ldots, L \\
    \mathbf{X}^{l} & = \mathbf{Z}^{l} + \text{MLP}(\text{LayerNorm}(\mathbf{Z}^{l})), \quad l = 1, \ldots, L \\
    \end{align*}

    This module outputs $\text{TransformerEncoder}(\mathbf{X}) = \mathbf{X}^{L}$ if `add_self=False`, 
    and $\text{TransformerEncoder}(\mathbf{X}) = \mathbf{X}^{L} + \mathbf{X}$ if `add_self=True`.

    """

    def __init__(
        self,
        in_dim: int,
        att_dim: int,
        n_heads: int = 4,
        n_layers: int = 4,
        use_mlp: bool = True,
        add_self: bool = False,
        dropout: float = 0.0
    ):
        """
        Class constructor

        Arguments:
            in_dim: Input dimension.
            att_dim: Attention dimension.
            n_heads: Number of heads.
            n_layers: Number of layers.
            use_mlp: Whether to use feedforward layer.
            add_self: Whether to add input to output.
            dropout: Dropout rate.        
        """        

        layers = torch.nn.ModuleList([TransformerLayer(
            att_dim, in_dim, n_heads, use_mlp=use_mlp, dropout=dropout) for _ in range(n_layers)])

        super(TransformerEncoder, self).__init__(layers, add_self=add_self)
        
        if in_dim != att_dim:
            self.in_proj = torch.nn.Linear(in_dim, att_dim)
            self.out_proj = torch.nn.Linear(att_dim, in_dim)
        else:
            self.in_proj = torch.nn.Identity()
            self.out_proj = torch.nn.Identity()
        
        self.norm = torch.nn.LayerNorm(att_dim)

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward method.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.
            mask: Mask tensor of shape `(batch_size, bag_size)`.

        Returns:
            Y: Output tensor of shape `(batch_size, bag_size, in_dim)`.        
        """

        X = self.in_proj(X)  # (batch_size, bag_size, att_dim)
        Y = super().forward(X, mask=mask)
        Y = self.norm(Y)
        Y = self.out_proj(Y)  # (batch_size, bag_size, in_dim)

        return Y


class SmTransformerEncoder(Encoder):
    r"""
    A Transformer encoder with the $\texttt{Sm}$ operator, skip connections and layer normalization.

    Given an input bag input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times D}$,
    it computes:

    \begin{align*}
    \mathbf{X}^{0} & = \mathbf{X} \\
    \mathbf{Z}^{l} & = \mathbf{X}^{l-1} + \texttt{Sm}( \text{SelfAttention}( \text{LayerNorm}(\mathbf{X}^{l-1}) ) ), \quad l = 1, \ldots, L \\
    \mathbf{X}^{l} & = \mathbf{Z}^{l} + \text{MLP}(\text{LayerNorm}(\mathbf{Z}^{l})), \quad l = 1, \ldots, L \\
    \end{align*}

    This module outputs $\text{SmTransformerEncoder}(\mathbf{X}) = \mathbf{X}^{L}$ if `add_self=False`,
    and $\text{SmTransformerEncoder}(\mathbf{X}) = \mathbf{X}^{L} + \mathbf{X}$ if `add_self=True`.
    """

    def __init__(
        self,
        in_dim: int,
        att_dim: int,
        n_heads: int = 4,
        n_layers: int = 4,
        use_mlp: bool = True,
        add_self: bool = False,
        dropout: float = 0.0,
        sm_alpha: float = None,
        sm_mode: str = None,
        sm_steps: int = 10
    ):
        """
        Class constructor

        Arguments:
            in_dim: Input dimension.
            att_dim: Attention dimension.
            n_heads: Number of heads.
            n_layers: Number of layers.
            use_mlp: Whether to use feedforward layer.
            add_self: Whether to add input to output.
            dropout: Dropout rate.        
        """

        layers = torch.nn.ModuleList([
            SmTransformerLayer(att_dim, in_dim, n_heads, use_mlp=use_mlp,
                                      dropout=dropout, sm_alpha=sm_alpha, sm_mode=sm_mode, sm_steps=sm_steps)
            for _ in range(n_layers)
        ])

        super(SmTransformerEncoder, self).__init__(layers, add_self=add_self)

        if in_dim != att_dim:
            self.in_proj = torch.nn.Linear(in_dim, att_dim)
            self.out_proj = torch.nn.Linear(att_dim, in_dim)
        else:
            self.in_proj = torch.nn.Identity()
            self.out_proj = torch.nn.Identity()
        self.norm = torch.nn.LayerNorm(att_dim)


    def forward(
        self,
        X: torch.Tensor,
        adj: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward method.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask tensor of shape `(batch_size, bag_size)`.

        Returns:
            Y: Output tensor of shape `(batch_size, bag_size, in_dim)`.        
        """

        X = self.in_proj(X)  # (batch_size, bag_size, att_dim)
        Y = super().forward(X, adj=adj, mask=mask)
        Y = self.norm(Y)
        Y = self.out_proj(Y)  # (batch_size, bag_size, in_dim)

        return Y

class NystromTransformerEncoder(Encoder):
    def __init__(
        self, 
        att_dim : int = 512,
        n_heads : int = 8,
        n_layers : int = 4,
        n_landmarks : int = 256,
        pinv_iterations : int = 6,
        residual : bool = True,
        dropout : float = 0.0,
        use_mlp : bool = False
    ) -> None:
        """
        Nystrom Transformer encoder.

        Arguments:
            att_dim: Attention dimension.
            n_heads: Number of heads.
            n_layers: Number of layers.
            n_landmarks: Number of landmarks.
            pinv_iterations: Number of iterations for the pseudo-inverse.
            residual: Whether to use residual in the attention layer.
            dropout: Dropout rate.
            use_mlp: Whether to use a MLP after the attention layer.   
        """
        layers = torch.nn.ModuleList([
            NystromTransformerLayer(
                att_dim=att_dim, n_heads=n_heads, n_landmarks=n_landmarks, pinv_iterations=pinv_iterations, residual=residual, dropout=dropout, use_mlp=use_mlp
            ) for _ in range(n_layers)
        ])

        super(NystromTransformerEncoder, self).__init__(layers, add_self=False)

        if att_dim != att_dim:
            self.in_proj = torch.nn.Linear(att_dim, att_dim)
            self.out_proj = torch.nn.Linear(att_dim, att_dim)
        else:
            self.in_proj = torch.nn.Identity()
            self.out_proj = torch.nn.Identity()
        
        self.norm = torch.nn.LayerNorm(att_dim)

    def forward(
        self,
        X: torch.Tensor,
        return_att: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, att_dim)`.
            return_att: Whether to return attention weights.

        Returns:
            Y: Output tensor of shape `(batch_size, bag_size, att_dim)`.
            att: Only returned when `return_att=True`. Attention weights of shape `(batch_size, n_heads, bag_size, bag_size)`.        
        """

        X = self.in_proj(X)  # (batch_size, bag_size, att_dim)
        Y = super().forward(X, return_att=return_att)
        Y = self.norm(Y)
        Y = self.out_proj(Y)

        return Y