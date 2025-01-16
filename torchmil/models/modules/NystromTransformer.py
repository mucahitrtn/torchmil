
import torch
from nystrom_attention import NystromAttention


class NystromTransformerLayer(torch.nn.Module):

    def __init__(self, dim=512, dim_head=64, heads=8):
        super().__init__()
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            num_landmarks = dim//2,     # number of landmarks
            pinv_iterations = 6,        # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.0
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            z, attn = self.attn(x, return_attn=True)
            x = x + z
            return x, attn
        else:
            x = x + self.attn(x)
            return x