
import torch
from nystrom_attention import NystromAttention


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
            return_attn: Whether to return attention weights.
        
        Returns:
            X: Output tensor of shape `(batch_size, bag_size, att_dim)`.
            att: Only returned when `return_att=True`. Attention weights of shape `(batch_size, n_heads, bag_size, bag_size)`.        
        """

        if return_att:
            X, att = self.attn(self.norm1(X), return_attn=True)
            if self.use_mlp:
                X = X + self.mlp(self.norm2(X))            
            return X, att
        else:
            X = X + self.attn(self.norm1(X))
            if self.use_mlp:
                X = X + self.mlp(self.norm2(X))
            return X