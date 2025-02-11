import torch

from torch.nn.attention import SDPBackend

from torchmil.models.modules import MultiheadSelfAttention, Sm

SDP_BACKEND = [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION,
               SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]


class TransformerEncoderLayer(torch.nn.Module):
    r"""
    One layer of the Transformer encoder.

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times D}$,
    this module computes:

    \begin{align*}
    \mathbf{Z} & = \mathbf{X} + \text{SelfAttention}( \text{LayerNorm}(\mathbf{X}) ) \\
    \mathbf{Y} & = \mathbf{Z} + \text{MLP}(\text{LayerNorm}(\mathbf{Z})), \\
    \end{align*}

    and outputs $\mathbf{Y}$.
    """

    def __init__(
        self,
        att_dim: int,
        in_dim: int = None,
        n_heads: int = 4,
        use_mlp: bool = True,
        dropout: float = 0.0
    ):
        """
        Class constructor.

        Arguments:
            att_dim: Attention dimension.
            in_dim: Input dimension. If None, in_dim = att_dim.
            n_heads: Number of heads.
            use_mlp: Whether to use feedforward layer.
            dropout: Dropout rate
        """
        super(TransformerEncoderLayer, self).__init__()
        self.att_dim = att_dim
        self.in_dim = in_dim
        if self.in_dim is None:
            self.in_dim = att_dim
        self.n_heads = n_heads
        self.use_mlp = use_mlp
        self.dropout = dropout

        self.mha_layer = MultiheadSelfAttention(
            att_dim, in_dim, n_heads, dropout=dropout)

        if self.use_mlp:
            self.mlp_layer = torch.nn.Sequential(
                torch.nn.Linear(att_dim, 4*att_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(4*att_dim, att_dim),
                torch.nn.Dropout(dropout)
            )

        self.norm1 = torch.nn.LayerNorm(att_dim)
        self.norm2 = torch.nn.LayerNorm(att_dim)

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

        # (batch_size, bag_size, in_dim)
        Y = X + self.mha_layer(self.norm1(X), mask=mask)
        if self.use_mlp:
            # (batch_size, bag_size, in_dim)
            Y = Y + self.mlp_layer(self.norm2(Y))
        return Y


class SmTransformerEncoderLayer(torch.nn.Module):
    r"""
    One layer of the Transformer encoder with the $\texttt{Sm}$ operator.

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times D}$,
    this module computes:

    \begin{align*}
    \mathbf{Z} & = \mathbf{X} + \texttt{Sm}( \text{SelfAttention}( \text{LayerNorm}(\mathbf{X}) ) )\\
    \mathbf{Y} & = \mathbf{Z} + \text{MLP}(\text{LayerNorm}(\mathbf{Z})), \\
    \end{align*}

    and outputs $\mathbf{Y}$.
    """

    def __init__(
        self,
        att_dim: int,
        in_dim: int = None,
        n_heads: int = 4,
        use_mlp: bool = True,
        dropout: float = 0.0,
        sm_alpha: float = None,
        sm_mode: str = None,
        sm_steps: int = 10
    ):
        """
        Class constructor.

        Arguments:
            att_dim: Attention dimension.
            in_dim: Input dimension. If None, in_dim = att_dim.
            n_heads: Number of heads.
            use_mlp: Whether to use feedforward layer.
            dropout: Dropout rate
            sm_alpha: Alpha value for the Sm operator.
            sm_mode: Sm mode.
            sm_steps: Number of steps to approximate the exact Sm operator.
        """
        super(SmTransformerEncoderLayer, self).__init__()
        if in_dim is None:
            in_dim = att_dim
        self.use_mlp = use_mlp

        self.mha_layer = MultiheadSelfAttention(
            att_dim, in_dim, n_heads, dropout=dropout)

        self.sm = Sm(alpha=sm_alpha, mode=sm_mode, num_steps=sm_steps)

        if self.use_mlp:
            self.mlp_layer = torch.nn.Sequential(
                torch.nn.Linear(att_dim, 4*att_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(4*att_dim, att_dim),
                torch.nn.Dropout(dropout)
            )
        else:
            self.mlp_layer = torch.nn.Identity()

        self.norm1 = torch.nn.LayerNorm(att_dim)
        self.norm2 = torch.nn.LayerNorm(att_dim)

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

        # (batch_size, bag_size, in_dim)
        Y = X + self.sm(self.mha_layer(self.norm1(X), mask=mask), adj)
        if self.use_mlp:
            # (batch_size, bag_size, in_dim)
            Y = Y + self.mlp_layer(self.norm2(Y))
        return Y


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

        layers = torch.nn.ModuleList([TransformerEncoderLayer(
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
    A Transformer encoder with the $\texttt{Sm} operator, skip connections and layer normalization.

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
            SmTransformerEncoderLayer(att_dim, in_dim, n_heads, use_mlp=use_mlp,
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
