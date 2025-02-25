import torch

class Layer(torch.nn.Module):
    """
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
        """
        Y = self.in_proj(X) + self.att_module(self.norm1(X), **kwargs)
        if self.use_mlp:
            Y = Y + self.mlp_module(self.norm2(Y))
        Y = self.out_proj(Y)
        return Y