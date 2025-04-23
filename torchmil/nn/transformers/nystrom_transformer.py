import torch

from torchmil.nn.attention import NystromAttention

from .encoder import Encoder
from .layer import Layer


class NystromTransformerLayer(Layer):
    r"""
    One layer of the NystromTransformer encoder.

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times D}$,
    this module computes:

    \begin{align*}
    \mathbf{Z} & = \mathbf{X} + \operatorname{NystromSelfAttention}( \operatorname{LayerNorm}(\mathbf{X}) ) \\
    \mathbf{Y} & = \mathbf{Z} + \operatorname{MLP}(\operatorname{LayerNorm}(\mathbf{Z})), \\
    \end{align*}

    and outputs $\mathbf{Y}$. $\operatorname{NystromSelfAttention}$ is implemented using the NystromAttention module, see [NystromAttention](../attention/nystrom_attention.md).
    """

    def __init__(
        self, 
        in_dim : int,
        out_dim : int = None,
        att_dim : int = 512,
        n_heads : int = 4,
        learn_weights : bool = True,
        n_landmarks : int = 256,
        pinv_iterations : int = 6,
        dropout : float = 0.0,
        use_mlp : bool = False
    ) -> None:
        """
        Arguments:
            att_dim: Attention dimension.
            n_heads: Number of heads.
            n_landmarks: Number of landmarks.
            pinv_iterations: Number of iterations for the pseudo-inverse.
            dropout: Dropout rate.
            use_mlp: Whether to use a MLP after the attention layer.   
        """
        att_module = NystromAttention(
            in_dim=in_dim, out_dim=out_dim, att_dim=att_dim, n_heads=n_heads, learn_weights=learn_weights, n_landmarks=n_landmarks, pinv_iterations=pinv_iterations
        )

        super(NystromTransformerLayer, self).__init__(
            in_dim=in_dim, out_dim=out_dim, att_dim=att_dim, att_module=att_module, use_mlp=use_mlp, dropout=dropout
        )

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

        return super().forward(X, return_att=return_att)


class NystromTransformerEncoder(Encoder):
    r"""
    Nystrom Transformer encoder with skip connections and layer normalization.

    Given an input bag input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times D}$,
    it computes:

    \begin{align*}
    \mathbf{X}^{0} & = \mathbf{X} \\
    \mathbf{Z}^{l} & = \mathbf{X}^{l-1} + \operatorname{NystromSelfAttention}( \operatorname{LayerNorm}(\mathbf{X}^{l-1}) ), \quad l = 1, \ldots, L \\
    \mathbf{X}^{l} & = \mathbf{Z}^{l} + \operatorname{MLP}(\operatorname{LayerNorm}(\mathbf{Z}^{l})), \quad l = 1, \ldots, L \\
    \end{align*}

    This module outputs $\operatorname{TransformerEncoder}(\mathbf{X}) = \mathbf{X}^{L}$ if `add_self=False`, 
    and $\operatorname{TransformerEncoder}(\mathbf{X}) = \mathbf{X}^{L} + \mathbf{X}$ if `add_self=True`.

    $\operatorname{NystromSelfAttention}$ is implemented using the NystromAttention module, see [NystromAttention](../attention/nystrom_attention.md).
    """

    def __init__(
        self, 
        in_dim : int,
        out_dim : int = None,
        att_dim : int = 512,
        n_heads : int = 8,
        n_layers : int = 4,
        n_landmarks : int = 256,
        pinv_iterations : int = 6,
        dropout : float = 0.0,
        use_mlp : bool = False
    ) -> None:
        """
        Arguments:
            att_dim: Attention dimension.
            n_heads: Number of heads.
            n_layers: Number of layers.
            n_landmarks: Number of landmarks.
            pinv_iterations: Number of iterations for the pseudo-inverse.
            dropout: Dropout rate.
            use_mlp: Whether to use a MLP after the attention layer.   
        """

        if out_dim is None:
            out_dim = in_dim

        layers = torch.nn.ModuleList([
            NystromTransformerLayer(
                att_dim=att_dim, in_dim=att_dim, out_dim=att_dim, n_heads=n_heads, n_landmarks=n_landmarks, pinv_iterations=pinv_iterations, dropout=dropout, use_mlp=use_mlp
            ) for _ in range(n_layers)
        ])

        super(NystromTransformerEncoder, self).__init__(layers, add_self=False)
        
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

        if return_att:
            Y, att = super().forward(X, return_att=True)
        else:
            Y = super().forward(X)
        Y = self.norm(Y)

        if return_att:
            return Y, att
        else:
            return Y