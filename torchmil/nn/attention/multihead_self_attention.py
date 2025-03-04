import torch

from torch.nn.attention import SDPBackend

SDP_BACKEND = [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]

class MultiheadSelfAttention(torch.nn.Module):
    """
    Multihead self-attention module.    
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
        Class constructor.

        Arguments:
            att_dim: Attention dimension.
            in_dim: Input dimension.
            out_dim: Output dimension. If None, out_dim = in_dim.
            n_heads: Number of heads.
            dropout: Dropout rate.
        """
        super(MultiheadSelfAttention, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.att_dim = att_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.head_dim = att_dim // n_heads
        self.learn_weights = learn_weights
        if learn_weights:
            self.qkv_nn = torch.nn.Linear(in_dim, 3 * att_dim, bias = False)
        else:
            self.qkv_nn = None
        
        if out_dim != att_dim:
            self.out_proj = torch.nn.Linear(att_dim, out_dim)
        else:
            self.out_proj = torch.nn.Identity()

    def _scaled_dot_product_attention(
            self, 
            query : torch.Tensor,
            key : torch.Tensor,
            value : torch.Tensor,
            mask : torch.Tensor = None,
            return_attention : bool = False
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Scaled dot product attention.

        Arguments:
            query: Query tensor of shape `(batch_size, n_heads, seq_len, head_dim)`.
            key: Key tensor of shape `(batch_size, n_heads, seq_len, head_dim)`.
            value: Value tensor of shape `(batch_size, n_heads, seq_len, head_dim)`.
            mask: Mask tensor of shape `(batch_size, seq_len)`.
            return_attention: Whether to return the attention matrices.
        Returns:
            out: Output tensor of shape `(batch_size, n_heads, seq_len, head_dim)`.                
        """

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, seq_len)
            mask = mask.repeat(1, 1, query.size(2), 1).bool() # (batch_size, n_heads, seq_len, seq_len)

        if return_attention:
            query = query / (self.head_dim ** 0.5)
            qk = torch.einsum("bqhd,bkhd->bhqk", query, key) # (batch_size, n_heads, seq_len, seq_len)
            if mask is not None:
                qk.masked_fill_(mask, float("-inf"))
            att = torch.nn.functional.softmax(qk, dim=-1)
            att_d = torch.nn.functional.dropout(att, p=self.dropout, training=self.training)
            out = torch.einsum("bhqk,bkhd->bqhd", att_d, value) # (batch_size, n_heads, seq_len, head_dim)
            return out, att
        else:
            with torch.nn.attention.sdpa_kernel(SDP_BACKEND):
                out = torch.nn.functional.scaled_dot_product_attention(query, key, value, mask, self.dropout) # (batch_size, n_heads, seq_len, head_dim)
            return out

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
            return_attention : bool = False
        ) -> torch.Tensor:
        """
        Forward pass.

        Arguments:
            x: Input tensor of shape `(batch_size, seq_len, in_dim)`.
            mask: Mask tensor of shape `(batch_size, seq_len)`.
        Returns:
            y: Output tensor of shape `(batch_size, seq_len, att_dim)`.
            att: Only returned when `return_attention=True`. Attention weights of shape `(batch_size, n_heads, seq_len, seq_len)`.
        """
        batch_size, seq_len, _ = x.size()
        query, key, value = self._qkv(x) # (batch_size, seq_len, att_dim), (batch_size, seq_len, att_dim), (batch_size, seq_len, att_dim)
        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, n_heads, seq_len, head_dim)
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, n_heads, seq_len, head_dim)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, n_heads, seq_len, head_dim)
        if return_attention:
            y, att = self._scaled_dot_product_attention(query, key, value, mask, return_attention)
            y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.att_dim) # (batch_size, seq_len, att_dim)
            y = self.out_proj(y)
            return y, att
        else:
            y = self._scaled_dot_product_attention(query, key, value, mask) # (batch_size, n_heads, seq_len, head_dim)
            y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.att_dim) # (batch_size, seq_len, att_dim)
            y = self.out_proj(y)
            return y