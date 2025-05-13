import torch
import math

from ..irpe import get_rpe_config, build_rpe

class iRPEMultiheadSelfAttention(torch.nn.Module):
    """
    Multihead Self-Attention with image Relative Position Encoding (iRPE), as described in [Rethinking and Improving Relative Position Encoding for Vision Transformer](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_Rethinking_and_Improving_Relative_Position_Encoding_for_Vision_Transformer_ICCV_2021_paper.html).

    The iRPE implementation is based on the [official codebase](https://github.com/microsoft/Cream/tree/main/iRPE).

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
        rpe_method : str = 'product',
        rpe_mode : str ='contextual',
        rpe_shared_head : bool = True,
        rpe_skip : int = 1,
        rpe_on : str = 'k',
    ):
        """
        Arguments:
            in_dim: Input dimension.
            att_dim: Attention dimension. Must be divisible by `n_heads`.
            out_dim: Output dimension. If None, `out_dim` = `in_dim`.
            n_heads: Number of heads.
            dropout: Dropout rate.
            learn_weights: If True, learn the weights for query, key, and value. If False, q, k, and v are the same as the input, and therefore `in_dim` must be divisible by `n_heads`.
            rpe_ratio: Relative position encoding ratio.
            rpe_method: Relative position encoding method. Possible values: ['euc', 'quant', 'cross', 'product']
            rpe_mode: Relative position encoding mode. Possible values: [None, 'bias', 'contextual']
            rpe_shared_head: Whether to share weights across heads.
            rpe_skip: Relative position encoding skip. Possible values: [0, 1]. 
            rpe_on: Where to apply relative positional encoding. Possible values: ['q', 'k', 'v', 'qk', 'kv', 'qkv'].

        **Note.** When 'v' is in `rpe_on`, `rpe_mode` must be 'contextual'.
        """
        super(iRPEMultiheadSelfAttention, self).__init__()

        if rpe_method not in ['euc', 'quant', 'cross', 'product']:
            raise ValueError("rpe_method must be one of ['euc', 'quant', 'cross', 'product']")

        if rpe_mode not in [None, 'bias', 'contextual']:
            raise ValueError("rpe_mode must be one of [None, 'bias', 'contextual']")
        
        if rpe_on not in ['q', 'k', 'v', 'qk', 'kv', 'qkv']:
            raise ValueError("rpe_on must be one of ['q', 'k', 'v', 'qk', 'kv', 'qkv']")
        
        if rpe_skip not in [0, 1]:
            raise ValueError("rpe_skip must be one of [0, 1]")
        
        if rpe_on not in ['q', 'k', 'v', 'qk', 'kv', 'qkv']:
            raise ValueError("rpe_on must be one of ['q', 'k', 'v', 'qk', 'kv', 'qkv']")

        if 'v' in rpe_on and rpe_mode != 'contextual':
            raise ValueError("When 'v' is in rpe_on, rpe_mode must be 'contextual'")

        if out_dim is None:
            out_dim = in_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.learn_weights = learn_weights
        self.rpe_skip = rpe_skip
        if learn_weights:
            self.qkv_nn = torch.nn.Linear(in_dim, 3 * att_dim, bias = False)
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
            return_att : bool = False
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
            return_att: Whether to return the attention matrices.

        Returns:
            out: Output tensor of shape `(batch_size, n_heads, seq_len, head_dim)`.
        """

        if mask is not None:
            # the following is equivalent to mask.unsqueeze(-1) @ mask.unsqueeze(-1).transpose(1, 2)
            mask = mask[:, None, :, None] * mask[:, None, None, :] # (batch_size, 1, seq_len, seq_len)
            mask = mask.bool() # (batch_size, 1, seq_len, seq_len)

        query = query / (self.head_dim ** 0.5)
        qk = torch.einsum("bhid,bhjd->bhij", query, key) # (batch_size, n_heads, seq_len, seq_len)

        if self.rpe_k is not None:
            qk += self.rpe_k(query, height=height, width=width)
        if self.rpe_q is not None:
            qk += self.rpe_q(key / (self.head_dim ** 0.5), height=height, width=width).transpose(2,3)

        if mask is not None:
            qk.masked_fill_(mask, float("-inf"))
            
        att = qk.softmax(dim=-1) # (batch_size, n_heads, seq_len, seq_len)
        att_d = torch.nn.functional.dropout(att, p=self.dropout, training=self.training)
        out = torch.einsum("bhij,bhjd->bhid", att_d, value) # (batch_size, n_heads, seq_len, head_dim)

        if self.rpe_v is not None:
            out += self.rpe_v(att_d, height=height, width=width)

        if return_att:
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
            return_att : bool = False,
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

        act_seq_len = seq_len - self.rpe_skip

        h = w = int(math.sqrt(act_seq_len))
        diff = act_seq_len - h * w
        if diff > 0:
            # if the sequence length is not a perfect square, we need to pad the input tensor so that rpe can be applied
            # compute the nearest perfect square greater than or equal to seq_len
            ps = math.ceil(math.sqrt(act_seq_len))**2
            new_seq_len = ps + self.rpe_skip
            padding = new_seq_len - act_seq_len
            x = torch.nn.functional.pad(x, (0, 0, 0, padding))
            if mask is not None:
                mask = torch.nn.functional.pad(mask, (0, padding), value=0)
        else:
            new_seq_len = seq_len

        query, key, value = self._qkv(x) # (batch_size, new_seq_len, att_dim), (batch_size, new_seq_len, att_dim), (batch_size, new_seq_len, att_dim)
        query = query.reshape(batch_size, self.n_heads, new_seq_len, -1) # (batch_size, n_heads, new_seq_len, head_dim)
        key = key.reshape(batch_size, self.n_heads, new_seq_len, -1) # (batch_size, n_heads, new_seq_len, head_dim)
        value = value.reshape(batch_size, self.n_heads, new_seq_len, -1) # (batch_size, n_heads, new_seq_len, head_dim)
        
        out = self._scaled_dot_product_attention(query, key, value, mask, height=height, width=width, return_att=return_att) # (batch_size, n_heads, new_seq_len, head_dim)

        if return_att:
            y = out[0]
            att = out[1]
        else:
            y = out

        y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, new_seq_len, self.att_dim) # (batch_size, new_seq_len, att_dim)
        y = self.out_proj(y)
        if diff > 0:
            y = y[:, :seq_len, :]
            if return_att:
                att = att[:, :, :seq_len, :seq_len]
        
        if return_att:
            return y, att
        else:
            return y
