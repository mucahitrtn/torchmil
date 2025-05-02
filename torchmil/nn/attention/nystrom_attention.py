from math import ceil
import torch
from einops import rearrange, reduce

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


class NystromAttention(torch.nn.Module):
    """
    Nystrom attention, as described in the paper [Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention](https://arxiv.org/abs/2102.03902).

    Implementation based on the [official code](https://github.com/lucidrains/nystrom-attention).
    """

    def __init__(
        self,
        in_dim : int,
        out_dim : int = None,
        att_dim : int = 512,
        n_heads : int = 4,
        learn_weights : bool = True,
        n_landmarks : int = 256,
        pinv_iterations : int = 6
    ):
        """
        Arguments:
            in_dim: Input dimension.
            out_dim: Output dimension. If None, out_dim = in_dim.
            att_dim: Attention dimension. Must be divisible by `n_heads`.
            n_heads: Number of heads.
            learn_weights: If True, learn the weights for query, key, and value. If False, q, k, and v are the same as the input, and therefore `in_dim` must be divisible by `n_heads`.
            n_landmarks: Number of landmarks.
            pinv_iterations: Number of iterations for Moore-Penrose pseudo-inverse.
        """
        super().__init__()

        if out_dim is None:
            out_dim = in_dim

        self.eps = 1e-8
        head_dim = att_dim // n_heads

        self.n_landmarks = n_landmarks
        self.pinv_iterations = pinv_iterations

        self.n_heads = n_heads
        self.scale_factor = head_dim ** -0.5
        self.learn_weights = learn_weights
        if learn_weights:
            self.qkv_nn = torch.nn.Linear(in_dim, att_dim * 3, bias = False)
            self.head_dim = att_dim // n_heads
            assert att_dim % n_heads == 0, "att_dim must be divisible by n_heads"
        else:
            self.qkv_nn = None
            self.head_dim = in_dim // n_heads
            att_dim = in_dim
            assert in_dim % n_heads == 0, "in_dim must be divisible by n_heads"

        if out_dim != att_dim:
            self.out_proj = torch.nn.Linear(att_dim, out_dim)
        else:
            self.out_proj = torch.nn.Identity()

    def _qkv(self, x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute query, key, and value tensors.

        Arguments:
            x: Input tensor of shape `(batch_size, seq_len, in_dim)`.
        Returns:
            query: Query tensor of shape `(batch_size, seq_len, att_dim)`.
            key: Key tensor of shape `(batch_size, seq_len, att_dim)`.
            value: Value tensor of shape `(batch_size, seq_len, att_dim)`.
        """
        if self.learn_weights:
            q, k, v = self.qkv_nn(x).chunk(3, dim=-1) # (batch_size, seq_len, att_dim), (batch_size, seq_len, att_dim), (batch_size, seq_len, att_dim)
        else:
            q = k = v = x
        return q, k, v

    def forward(
        self,
        x : torch.Tensor,
        mask : torch.Tensor = None,
        return_att : bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Arguments:
            x: Input tensor of shape `(batch_size, seq_len, in_dim)`.
            mask: Mask tensor of shape `(batch_size, seq_len)`.
            return_att: Whether to return attention weights.

        Returns:
            y: Output tensor of shape `(batch_size, seq_len, att_dim)`.
            att: Only returned when `return_att=True`. Attention weights of shape `(batch_size, n_heads, seq_len, seq_len)`.
        """

        batch_size, seq_len, _ = x.size()

        # m = num_landmarks, h = heads, d = head_dim, iters = pinv_iterations, eps = eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = seq_len % self.n_landmarks
        if remainder > 0:
            padding = self.n_landmarks - remainder
            x = torch.nn.functional.pad(x, (0, 0, padding, 0), value = 0) # (batch_size, seq_len + padding, att_dim)

            if mask is not None:
                mask = torch.nn.functional.pad(mask, (padding, 0), value = False) # (batch_size, seq_len + padding)
        new_seq_len = x.size(1)
        q, k, v = self._qkv(x) # (batch_size, new_seq_len, att_dim), (batch_size, new_seq_len, att_dim), (batch_size, new_seq_len, att_dim)
        q = q.view(batch_size, new_seq_len, self.n_heads, -1).permute(0, 2, 1, 3) # (batch_size, n_heads, new_seq_len, head_dim)
        k = k.view(batch_size, new_seq_len, self.n_heads, -1).permute(0, 2, 1, 3) # (batch_size, n_heads, new_seq_len, head_dim)
        v = v.view(batch_size, new_seq_len, self.n_heads, -1).permute(0, 2, 1, 3) # (batch_size, n_heads, new_seq_len, head_dim)

        # set masked positions to 0 in queries, keys, values

        if mask is not None:
            mask = mask[:, None, :]
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale_factor

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(seq_len / self.n_landmarks)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l) # (batch_size, n_heads, n_landmarks, head_dim)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l) # (batch_size, n_heads, n_landmarks, head_dim)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if mask is not None:
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + self.eps
            mask_landmarks = mask_landmarks_sum > 0

        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = torch.einsum(einops_eq, q, k_landmarks)
        sim2 = torch.einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = torch.einsum(einops_eq, q_landmarks, k)

        # masking

        if mask is not None:
            mask_value = -torch.finfo(q.dtype).max
            sim1= sim1.masked_fill(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2 = sim2.masked_fill(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3 = sim3.masked_fill(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1 = sim1.softmax(dim = -1)
        attn2 = sim2.softmax(dim = -1)
        attn3 = sim3.softmax(dim = -1)

        attn2_inv = moore_penrose_iter_pinv(attn2, self.pinv_iterations)

        out = (attn1 @ attn2_inv) @ (attn3 @ v) # (batch_size, n_heads, new_seq_len, head_dim)

        # add depth-wise conv residual of values

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h = self.n_heads) # (batch_size, new_seq_len, att_dim)
        out = self.out_proj(out) # (batch_size, new_seq_len, att_dim)
        out = out[:, -seq_len:] # remove padding

        if return_att:
            attn = attn1 @ attn2_inv @ attn3
            # remove padding
            attn = attn[:, :, -seq_len:, -seq_len:]
            return out, attn
        else:
            return out
