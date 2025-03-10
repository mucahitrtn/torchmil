import torch

from torchmil.nn.attention import NystromAttention

from .encoder import Encoder
from .layer import Layer


class NystromTransformerLayer(Layer):
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

        Nystrom Transformer layer.

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
    def __init__(
        self, 
        in_dim : int,
        out_dim : int = None,
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

        if out_dim is None:
            out_dim = in_dim

        layers = torch.nn.ModuleList([
            NystromTransformerLayer(
                att_dim=att_dim, in_dim=att_dim, out_dim=att_dim, n_heads=n_heads, n_landmarks=n_landmarks, pinv_iterations=pinv_iterations, residual=residual, dropout=dropout, use_mlp=use_mlp
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