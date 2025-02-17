import torch

from torch.nn.attention import SDPBackend

SDP_BACKEND = [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]

class MultiheadCrossAttention(torch.nn.Module):
    """
    Multihead self-attention module.    
    """
    def __init__(
            self, 
            att_dim : int, 
            in_dim : int = None,
            num_heads : int = 4,
            dropout : float = 0.0,
            learn_weights : bool = True
        ):
        """
        Class constructor.

        Arguments:
            att_dim: Attention dimension.
            in_dim: Input dimension.
            num_heads: Number of heads.
            dropout: Dropout rate.
        """
        super(MultiheadCrossAttention, self).__init__()
        self.att_dim = att_dim
        self.in_dim = in_dim
        if self.in_dim is None:
            self.in_dim = att_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = self.att_dim // num_heads
        self.learn_weights = learn_weights
        if learn_weights:
            self.q_nn = torch.nn.Linear(self.in_dim, self.att_dim, bias = False)
            self.kv_nn = torch.nn.Linear(self.in_dim, 2 * self.att_dim, bias = False)
        else:
            self.qkv_nn = None

    def _scaled_dot_product_attention(
            self, 
            query : torch.Tensor,
            key : torch.Tensor,
            value : torch.Tensor,
            mask : torch.Tensor = None
        ) -> torch.Tensor:
        """

        Scaled dot product attention.

        Arguments:
            query: Query tensor of shape `(batch_size, num_heads, seq_len, head_dim)`.
            key: Key tensor of shape `(batch_size, num_heads, seq_len, head_dim)`.
            value: Value tensor of shape `(batch_size, num_heads, seq_len, head_dim)`.
            mask: Mask tensor of shape `(batch_size, seq_len)`.
        Returns:
            out: Output tensor of shape `(batch_size, num_heads, seq_len, head_dim)`.                
        """

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, seq_len)
            mask = mask.repeat(1, 1, query.size(2), 1).bool() # (batch_size, num_heads, seq_len, seq_len)
      
        with torch.nn.attention.sdpa_kernel(SDP_BACKEND):
            out = torch.nn.functional.scaled_dot_product_attention(query, key, value, mask, self.dropout) # (batch_size, num_heads, seq_len, head_dim)
        
        return out

    def _qkv(self, x : torch.Tensor, u : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute query, key, and value tensors.

        Arguments:
            x: Input tensor of shape `(batch_size, seq_len_x, in_dim)`.
            u: Input tensor of shape `(batch_size, seq_len_u, in_dim)`.
        Returns:
            query: Query tensor of shape `(batch_size, seq_len, att_dim)`.
            key: Key tensor of shape `(batch_size, seq_len, att_dim)`.
            value: Value tensor of shape `(batch_size, seq_len, att_dim)`.        
        """
        if self.learn_weights:
            q = self.q_nn(x) # (batch_size, seq_len_x, att_dim)
            k, v = self.kv_nn(u).chunk(2, dim=-1) # (batch_size, seq_len_u, att_dim), (batch_size, seq_len_u, att_dim)
        else:
            q = x
            k = v = u
        return q, k, v

    def forward(
            self, 
            x : torch.Tensor,
            u : torch.Tensor,
            mask : torch.Tensor = None
        ) -> torch.Tensor:
        """
        Forward pass.

        Arguments:
            x: Input tensor of shape `(batch_size, seq_len_x, in_dim)`.
            u: Input tensor of shape `(batch_size, seq_len_u, in_dim)`.
            mask: Mask tensor of shape `(batch_size, seq_len)`.
        Returns:
            y: Output tensor of shape `(batch_size, seq_len_u, att_dim)`.
        """
        batch_size, seq_len_x, _ = x.size()
        seq_len_u = u.size(1)
        query, key, value = self._qkv(x, u) # (batch_size, seq_len_x, att_dim), (batch_size, seq_len_u, att_dim), (batch_size, seq_len_u, att_dim)
        query = query.view(batch_size, seq_len_x, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len_x, head_dim)
        key = key.view(batch_size, seq_len_u, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len_u, head_dim)
        value = value.view(batch_size, seq_len_u, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len_u, head_dim)
        y = self._scaled_dot_product_attention(query, key, value, mask) # (batch_size, num_heads, seq_len_u, head_dim)
        y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len_x, self.att_dim) # (batch_size, seq_len_u, att_dim)
        
        return y