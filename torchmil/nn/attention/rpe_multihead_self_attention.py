import torch

from ..irpe import get_rpe_config, build_rpe

class RPEMultiheadSelfAttention(torch.nn.Module):
    """
    Multihead Self-Attention with Relative Position Encoding (RPE), as described in [Rethinking and Improving Relative Position Encoding for Vision Transformer](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_Rethinking_and_Improving_Relative_Position_Encoding_for_Vision_Transformer_ICCV_2021_paper.html).
    
    The RPE implementation is based on the [official codebase](https://github.com/microsoft/Cream/tree/main/iRPE).
    
    """
    def __init__(
        self, 
        in_dim : int,
        out_dim : int = None,
        att_dim : int = 512,
        n_heads : int = 4,
        dropout : float = 0.0,
        learn_weights : bool = True,
        rpe_ratio : float = 1.9,
        rpe_method : str = "product",
        rpe_mode : str = 'ctx',
        rpe_shared_head : bool = True,
        rpe_skip : int = 1,
        rpe_on : str = 'k',
    ):
        """
        Arguments:
            in_dim: Input dimension.
            att_dim: Attention dimension.
            out_dim: Output dimension. If None, out_dim = in_dim.
            n_heads: Number of heads.
            dropout: Dropout rate.
            learn_weights: Whether to learn the query, key, and value weights.
            rpe_ratio: Relative position encoding ratio.
            rpe_method: Relative position encoding method.
            rpe_mode: Relative position encoding mode.
            rpe_shared_head: Whether to share relative position encoding weights across heads.
            rpe_skip: Relative position encoding skip.
            rpe_on: Relative position encoding on query, key, or value.
        """
        super(RPEMultiheadSelfAttention, self).__init__()
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
        
        rpe_config = get_rpe_config(
            ratio=rpe_ratio,
            method=rpe_method,
            mode=rpe_mode,
            shared_head=rpe_shared_head,
            skip=rpe_skip,
            rpe_on=rpe_on
        )

        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config, head_dim=self.head_dim, num_heads=n_heads)

    def _scaled_dot_product_attention(
            self, 
            query : torch.Tensor,
            key : torch.Tensor,
            value : torch.Tensor,
            mask : torch.Tensor = None,
            height : int = None,
            width : int = None,
            return_attention : bool = False
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Scaled dot product attention.

        Arguments:
            query: Query tensor of shape `(batch_size, n_heads, seq_len, head_dim)`.
            key: Key tensor of shape `(batch_size, n_heads, seq_len, head_dim)`.
            value: Value tensor of shape `(batch_size, n_heads, seq_len, head_dim)`.
            mask: Mask tensor of shape `(batch_size, seq_len)`.
            height: Height of the input sequence. If None, `height = floor(sqrt(seq_len))`.
            width: Width of the input sequence. If None, `width = floor(sqrt(seq_len)`).
            return_attention: Whether to return the attention matrices.

        Returns:
            out: Output tensor of shape `(batch_size, n_heads, seq_len, head_dim)`.                
        """

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, seq_len)
            mask = mask.repeat(1, 1, query.size(2), 1).bool() # (batch_size, n_heads, seq_len, seq_len)
        
        query = query / (self.head_dim ** 0.5)
        qk = torch.einsum("bhid,bhjd->bhij", query, key) # (batch_size, n_heads, seq_len, seq_len)

        if self.rpe_k is not None:
            qk += self.rpe_k(query, height=height, width=width)
        if self.rpe_q is not None:
            qk += self.rpe_q(key / (self.head_dim ** 0.5), height=height, width=width).transpose(2,3)

        if mask is not None:
            qk.masked_fill_(mask, float("-inf"))
        att = torch.nn.functional.softmax(qk, dim=-1)
        att_d = torch.nn.functional.dropout(att, p=self.dropout, training=self.training)
        out = torch.einsum("bhij,bhjd->bhid", att_d, value) # (batch_size, n_heads, seq_len, head_dim)

        if self.rpe_v is not None:
            out += self.rpe_v(att_d, height=height, width=width)
        
        if return_attention:
            return out, att
        else:
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
            return_attention : bool = False,
            height : int = None,
            width : int = None
        ) -> torch.Tensor:
        """
        Forward pass.

        Arguments:
            x: Input tensor of shape `(batch_size, seq_len, in_dim)`.
            mask: Mask tensor of shape `(batch_size, seq_len)`.
            height: Height of the input sequence. If None, `height = floor(sqrt(seq_len))`.
            width: Width of the input sequence. If None, `width = floor(sqrt(seq_len))`.

        Returns:
            y: Output tensor of shape `(batch_size, seq_len, att_dim)`.
        """
        batch_size, seq_len, _ = x.size()
        query, key, value = self._qkv(x) # (batch_size, seq_len, att_dim), (batch_size, seq_len, att_dim), (batch_size, seq_len, att_dim)
        query = query.view(batch_size, seq_len, self.n_heads, -1).permute(0, 2, 1, 3) # (batch_size, n_heads, seq_len, head_dim)
        key = key.view(batch_size, seq_len, self.n_heads, -1).permute(0, 2, 1, 3) # (batch_size, n_heads, seq_len, head_dim)
        value = value.view(batch_size, seq_len, self.n_heads, -1).permute(0, 2, 1, 3) # (batch_size, n_heads, seq_len, head_dim)
        if return_attention:
            y, att = self._scaled_dot_product_attention(query, key, value, mask, height=height, width=width, return_attention=True) # (batch_size, n_heads, seq_len, head_dim)
            y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.att_dim) # (batch_size, seq_len, att_dim)
            y = self.out_proj(y)
            return y, att
        else:
            y = self._scaled_dot_product_attention(query, key, value, mask) # (batch_size, n_heads, seq_len, head_dim)
            y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.att_dim) # (batch_size, seq_len, att_dim)
            y = self.out_proj(y)
            return y