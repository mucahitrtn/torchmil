import torch

from torch.nn.attention import SDPBackend

from torchmil.nn import MultiheadSelfAttention, Sm

from nystrom_attention import NystromAttention


SDP_BACKEND = [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION,
               SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]


class TransformerLayer(torch.nn.Module):
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
        super(TransformerLayer, self).__init__()
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


class SmTransformerLayer(torch.nn.Module):
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
        super(SmTransformerLayer, self).__init__()
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

class NystromTransformerLayer(torch.nn.Module):
    def __init__(
        self, 
        att_dim : int = 512,
        n_heads : int = 8,
        n_landmarks : int = 256,
        pinv_iterations : int = 6,
        residual : bool = True,
        dropout : float = 0.0,
        use_mlp : bool = False
    ) -> None:
        """

        Nystrom Transformer layer.

        Arguments:
            att_dim: Attention dimension.
            n_heads: Number of heads.
            n_landmarks: Number of landmarks.
            pinv_iterations: Number of iterations for the pseudo-inverse.
            residual: Whether to use residual in the attention layer.
            dropout: Dropout rate.
            use_mlp: Whether to use a MLP after the attention layer.   
        """
        super().__init__()
        self.use_mlp = use_mlp
        self.attn = NystromAttention(
            dim = att_dim,
            dim_head = att_dim // n_heads,
            heads = n_heads,
            num_landmarks = n_landmarks,
            pinv_iterations = pinv_iterations,
            residual = residual,
            dropout = dropout
        )

        if use_mlp:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(att_dim, 4*att_dim),
                torch.nn.GELU(),
                torch.nn.Linear(4*att_dim, att_dim)
            )
        
        self.norm1 = torch.nn.LayerNorm(att_dim)
        self.norm2 = torch.nn.LayerNorm(att_dim)

    def forward(
            self, 
            X : torch.Tensor,
            return_att : bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, att_dim)`.
            return_att: Whether to return attention weights.
        
        Returns:
            X: Output tensor of shape `(batch_size, bag_size, att_dim)`.
            att: Only returned when `return_att=True`. Attention weights of shape `(batch_size, n_heads, bag_size, bag_size)`.        
        """

        if return_att:
            X, att = self.attn(self.norm1(X), return_att=True)
            if self.use_mlp:
                X = X + self.mlp(self.norm2(X))            
            return X, att
        else:
            X = X + self.attn(self.norm1(X))
            if self.use_mlp:
                X = X + self.mlp(self.norm2(X))
            return X