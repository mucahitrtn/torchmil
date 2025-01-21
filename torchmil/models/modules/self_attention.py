import torch

from torch.nn.attention import SDPBackend

SDP_BACKEND = [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]

class MultiheadSelfAttention(torch.nn.Module):
    """
    Multihead self-attention module.    
    """
    def __init__(
            self, 
            att_dim : int = 128, 
            in_dim : int = None,
            num_heads : int = 4,
            dropout : float = 0.0
        ):
        """
        Class constructor.

        Arguments:
            att_dim: Attention dimension.
            in_dim: Input dimension.
            num_heads: Number of heads.
            dropout: Dropout rate.        
        """
        super(MultiheadSelfAttention, self).__init__()
        self.att_dim = att_dim
        self.in_dim = in_dim
        if self.in_dim is None:
            self.in_dim = att_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = self.att_dim // num_heads
        self.qkv_nn = torch.nn.Linear(self.in_dim, 3 * self.att_dim, bias = False)

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
        q, k, v = self.qkv_nn(x).chunk(3, dim=-1) # (batch_size, seq_len, att_dim), (batch_size, seq_len, att_dim), (batch_size, seq_len, att_dim)
        return q, k, v

    def forward(
            self, 
            x : torch.Tensor,
            mask : torch.Tensor = None
        ) -> torch.Tensor:
        """
        Forward pass.

        Arguments:
            x: Input tensor of shape `(batch_size, seq_len, in_dim)`.
            mask: Mask tensor of shape `(batch_size, seq_len)`.
        Returns:
            y: Output tensor of shape `(batch_size, seq_len, att_dim)`.
        """
        batch_size, seq_len, _ = x.size()
        query, key, value = self._qkv(x) # (batch_size, seq_len, att_dim), (batch_size, seq_len, att_dim), (batch_size, seq_len, att_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_dim)
        y = self._scaled_dot_product_attention(query, key, value, mask) # (batch_size, num_heads, seq_len, head_dim)
        y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.att_dim) # (batch_size, seq_len, att_dim)
        
        return y