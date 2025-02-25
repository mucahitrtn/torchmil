import torch

from nystrom_attention import NystromAttention

from .encoder import Encoder


class NystromTransformerLayer(torch.nn.Module):
    def __init__(
        self, 
        in_dim : int,
        out_dim : int = None,
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

        if out_dim is None:
            out_dim = in_dim

        self.use_mlp = use_mlp
        self.attn = NystromAttention(
            dim = in_dim,
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
                torch.nn.Dropout(dropout),
                torch.nn.Linear(4*att_dim, att_dim),
                torch.nn.Dropout(dropout),
            )
        
        if out_dim != att_dim:
            self.proj_out = torch.nn.Linear(att_dim, out_dim)
        else:
            self.proj_out = torch.nn.Identity()
        
        self.norm1 = torch.nn.LayerNorm(in_dim)
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
            X = self.proj_out(X)     
            return X, att
        else:
            X = self.attn(self.norm1(X))
            if self.use_mlp:
                X = X + self.mlp(self.norm2(X))
            X = self.proj_out(X)
            return X


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

        if in_dim != att_dim:
            self.in_proj = torch.nn.Linear(in_dim, att_dim)
        else:
            self.in_proj = torch.nn.Identity()
        
        if out_dim != att_dim:
            self.out_proj = torch.nn.Linear(att_dim, out_dim)
        else:
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
        if return_att:
            Y, att = super().forward(X, return_att=True)
        else:
            Y = super().forward(X)
        Y = self.norm(Y)
        Y = self.out_proj(Y)

        if return_att:
            return Y, att
        else:
            return Y