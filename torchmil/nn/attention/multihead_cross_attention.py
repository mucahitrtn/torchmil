import torch

from torch.nn.attention import SDPBackend

SDP_BACKEND = [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]

class MultiheadCrossAttention(torch.nn.Module):
    r"""
    The Multihead Cross Attention module, as described in [Attention is All You Need](https://arxiv.org/abs/1706.03762).

    Given input bags $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times \texttt{in_dim}}$,
    and $\mathbf{Y} = \left[ \mathbf{y}_1, \ldots, \mathbf{y}_M \right]^\top \in \mathbb{R}^{M \times \texttt{in_dim}}$,
    this module computes:

    \begin{gather*}
    \mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{Y}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{Y}\mathbf{W}_V,\\
    \mathbf{Z} = \operatorname{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d}} \right) \mathbf{V},
    \end{gather*}

    where $d = \texttt{att_dim}$ and $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{\texttt{in_dim} \times \texttt{att_dim}}$ are learnable weight matrices.

    If $\texttt{out_dim} \neq \texttt{att_dim}$, $\mathbf{Y}$ is passed through a linear layer with output dimension $\texttt{out_dim}$.
    """
    def __init__(
            self,
            in_dim : int,
            out_dim : int = None,
            att_dim : int = 512,
            n_heads : int = 4,
            dropout : float = 0.0,
            learn_weights : bool = True
        ):
        """
        Arguments:
            in_dim: Input dimension.
            out_dim: Output dimension. If None, `out_dim` = `in_dim`.
            att_dim: Attention dimension, must be divisible by `n_heads`.
            n_heads: Number of heads.
            dropout: Dropout rate.
            learn_weights: If True, learn the weights for query, key, and value. If False, q, k, and v are the same as the input, and therefore `in_dim` must be divisible by `n_heads`.
        """
        super(MultiheadCrossAttention, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.in_dim = in_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.learn_weights = learn_weights
        if learn_weights:
            self.q_nn = torch.nn.Linear(self.in_dim, att_dim, bias = False)
            self.kv_nn = torch.nn.Linear(self.in_dim, 2 * att_dim, bias = False)
            self.head_dim = att_dim // n_heads
            assert att_dim % n_heads == 0, "att_dim must be divisible by n_heads"
        else:
            self.qkv_nn = None
            self.head_dim = in_dim // n_heads
            att_dim = in_dim
            assert in_dim % n_heads == 0, "in_dim must be divisible by n_heads"
        self.att_dim = att_dim

        if out_dim != att_dim:
            self.out_proj = torch.nn.Linear(att_dim, out_dim)
        else:
            self.out_proj = torch.nn.Identity()

    def _scaled_dot_product_attention(
            self,
            query : torch.Tensor,
            key : torch.Tensor,
            value : torch.Tensor,
            mask_query : torch.Tensor = None,
        ) -> torch.Tensor:
        """

        Scaled dot product attention.

        Arguments:
            query: Query tensor of shape `(batch_size, n_heads, seq_len_q, head_dim)`.
            key: Key tensor of shape `(batch_size, n_heads, seq_len_k, head_dim)`.
            value: Value tensor of shape `(batch_size, n_heads, seq_len_v, head_dim)`.
            mask_query: Mask tensor of shape `(batch_size, seq_len_q)`.
        Returns:
            out: Output tensor of shape `(batch_size, n_heads, seq_len, head_dim)`.
        """

        if mask_query is not None:
            mask_key = torch.ones(query.size(0), query.size(2), device=query.device) # (batch_size, seq_len_k)
            # the following is equivalent to mask.unsqueeze(-1) @ mask.unsqueeze(-1).transpose(1, 2)
            mask = mask_key.unsqueeze(-1).int() @ mask_query.unsqueeze(-1).transpose(1, 2).int() # (batch_size, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1).bool() # (batch_size, 1, seq_len_q, seq_len_k)
        else:
            mask = None

        with torch.nn.attention.sdpa_kernel(SDP_BACKEND):
            out = torch.nn.functional.scaled_dot_product_attention(query, key, value, mask, self.dropout) # (batch_size, n_heads, seq_len, head_dim)

        return out

    def _qkv(self, x : torch.Tensor, y : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute query, key, and value tensors.

        Arguments:
            x: Input tensor of shape `(batch_size, seq_len_x, in_dim)`.
            y: Input tensor of shape `(batch_size, seq_len_y, in_dim)`.
        Returns:
            query: Query tensor of shape `(batch_size, seq_len, att_dim)`.
            key: Key tensor of shape `(batch_size, seq_len, att_dim)`.
            value: Value tensor of shape `(batch_size, seq_len, att_dim)`.
        """
        if self.learn_weights:
            q = self.q_nn(x) # (batch_size, seq_len_x, att_dim)
            k, v = self.kv_nn(y).chunk(2, dim=-1) # (batch_size, seq_len_y, att_dim), (batch_size, seq_len_y, att_dim)
        else:
            q = x
            k = v = y
        return q, k, v

    def forward(
            self,
            x : torch.Tensor,
            y : torch.Tensor,
            mask : torch.Tensor = None,
        ) -> torch.Tensor:
        """
        Forward pass.

        Arguments:
            x: Input tensor of shape `(batch_size, seq_len_x, in_dim)`.
            y: Input tensor of shape `(batch_size, seq_len_y, in_dim)`.
            mask: Mask tensor of shape `(batch_size, seq_len_x)`.
        Returns:
            y: Output tensor of shape `(batch_size, seq_len_x, att_dim)`.
        """
        batch_size, seq_len_x, _ = x.size()
        seq_len_y = y.size(1)
        query, key, value = self._qkv(x, y) # (batch_size, seq_len_x, att_dim), (batch_size, seq_len_y, att_dim), (batch_size, seq_len_y, att_dim)
        query = query.view(batch_size, seq_len_x, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, n_heads, seq_len_x, head_dim)
        key = key.view(batch_size, seq_len_y, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, n_heads, seq_len_y, head_dim)
        value = value.view(batch_size, seq_len_y, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, n_heads, seq_len_y, head_dim)
        y = self._scaled_dot_product_attention(query, key, value, mask) # (batch_size, n_heads, seq_len_y, head_dim)
        y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len_x, self.att_dim) # (batch_size, seq_len_y, att_dim)
        y = self.out_proj(y)
        return y
