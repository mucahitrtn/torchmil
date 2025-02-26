import torch
from einops import rearrange

# from . import (
#     RelPropClone, 
#     RelPropAdd, 
#     RelPropIdentity, 
#     RelPropLinear, 
#     RelPropLayerNorm, 
#     RelPropSequential, 
#     RelPropGELU, 
#     RelPropMultiheadSelfAttention
# )

def forward_hook(module, input, output):
    ctx = {}
    
    X = []
    for arg in input:
        if torch.is_tensor(arg):
            X.append(arg.detach().requires_grad_(True))
        else:
            X.append(arg)
    ctx['X'] = X
    module.ctx = ctx

def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())

class RelPropModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx = None
        # self.register_module_forward_hook(forward_hook)
        torch.nn.modules.module.register_module_forward_hook(forward_hook, always_call=True)
    
    # def _register_context(self, X):
    #     if torch.is_tensor(X):
    #         self.ctx = {'X': X.detach().requires_grad_(True)}
    #     elif type(X) in (list, tuple):
    #         self.ctx = {'X': [x.detach().requires_grad_(True) if torch.is_tensor(x) else x for x in X]}
    #     else:
    #         raise ValueError(f"Unsupported input type {type(X)}")
    
    def _jvp(self, Y, X, Z):
        """
        Compute the Jacobian-vector product J_{X}(Y) Zs.

        Arguments:
            Z:
            X: 
            Y:
        """
        return torch.autograd.grad(Y, X, Z, retain_graph=True)

    def _relprop(self, ctx, R, **kwargs):
        raise NotImplementedError
    
    def relprop(self, R, **kwargs):
        return self._relprop(self.ctx, R, **kwargs)

class SimpleRelPropModule(RelPropModule):
    def _relprop(self, ctx, R, **kwargs):
        return R

class TaylorRelPropModule(RelPropModule):
    def _relprop(self, ctx, R, **kwargs):
        X = ctx['X']
        # Y = ctx['Y']
        Y = self.forward(X)
        S = safe_divide(R, Y)
        C = self._jvp(Y, X, S)

        if not torch.is_tensor(X):
            out = [ X[i]*C[i] for i in range(len(C)) ]
        else:
            out = X * C
        return out

class RelPropReLU(torch.nn.ReLU, SimpleRelPropModule):
    pass
    
class RelPropSoftmax(torch.nn.Softmax, SimpleRelPropModule):
    pass
    
class RelPropLayerNorm(torch.nn.LayerNorm, SimpleRelPropModule):
    pass

class RelPropDropout(torch.nn.Dropout, SimpleRelPropModule):
    pass

class RelPropIdentity(torch.nn.Identity, SimpleRelPropModule):
    pass

class RelPropClone(RelPropModule):
    def forward(self, X, n):
        return [X for _ in range(n)]

    def _relprop(self, ctx, R, **kwargs):
        X, n = ctx['X']
        Y = self.forward(X, n)
        S = [safe_divide(r, y) for r, y in zip(R, Y)]
        C = self._jvp(Y, X, S)

        R = X * C[0]

        return R

class RelPropAdd2(RelPropModule):
    def forward(self, X1, X2):
        return torch.add(X1, X2)

    def _relprop(self, ctx, R, **kwargs):
        X = ctx['X']
        # Y = ctx['Y']
        Y = self.forward(*X)
        S = safe_divide(R, Y)
        C = self._jvp(Y, X, S)

        X1, X2 = X
        C1, C2 = C

        a = X1 * C1 
        b = X2 * C2

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())

        return a, b

class RelPropLinear(torch.nn.Linear, RelPropModule):

    def _compute_relevances(self, R, w1, w2, x1, x2):
        Z1 = torch.nn.functional.linear(x1, w1)
        Z2 = torch.nn.functional.linear(x2, w2)
        S1 = safe_divide(R, Z1 + Z2)
        S2 = safe_divide(R, Z1 + Z2)
        C1 = x1 * torch.autograd.grad(Z1, x1, S1)[0]
        C2 = x2 * torch.autograd.grad(Z2, x2, S2)[0]
        return C1 + C2


    def _relprop(self, ctx, R, alpha=0.5, **kwargs):
        X = ctx['X'][0]
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(X, min=0)
        nx = torch.clamp(X, max=0)
            
        activator_relevances = self._compute_relevances(R, pw, nw, px, nx)
        inhibitor_relevances = self._compute_relevances(R, nw, pw, px, nx)

        R = alpha * activator_relevances + (1-alpha) * inhibitor_relevances

        return R

class RelPropEinsum(RelPropModule):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation

    def forward(self, *args):
        return torch.einsum(self.equation, *args)

    def _relprop(self, ctx, R, **kwargs):
        X = ctx['X']
        Y = self.forward(*X)
        S = safe_divide(R, Y)
        C = self._jvp(Y, X, S)
        return C

class RelPropSequential(torch.nn.Sequential, RelPropModule):
    def _relprop(self, ctx, R, **kwargs):
        for module in reversed(self):
            R = module.relprop(R, **kwargs)
        return R

class RelPropMultiheadSelfAttention(RelPropModule):
    r"""
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
        Class constructor

        Arguments:
            att_dim: Attention dimension.
            in_dim: Input dimension.
            out_dim: Output dimension. If None, out_dim = in_dim.
            num_heads: Number of heads.
            dropout: Dropout rate.
        """

        super().__init__()

        if out_dim is None:
            out_dim = in_dim

        self.att_dim = att_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.head_dim = att_dim // n_heads
        self.learn_weights = learn_weights
        if learn_weights:
            self.qkv_nn = RelPropLinear(in_dim, 3 * att_dim, bias=False)
        else:
            self.qkv_nn = RelPropIdentity()
        
        if out_dim != att_dim:
            self.out_proj = RelPropLinear(att_dim, out_dim)
        else:
            self.out_proj = RelPropIdentity()
        
        self.softmax = RelPropSoftmax(dim=-1)
        self.dropout = RelPropDropout(dropout)

        self.matmul1 = RelPropEinsum("b h i d, b h j d -> b h i j")
        self.matmul2 = RelPropEinsum("b h i j, b h j d -> b h i d")

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        """
        QKV = self.qkv_nn(X) # (batch_size, seq_len, 3 * att_dim)
        Q, K, V = rearrange(QKV, 'b n (p h d) -> p b h n d', h=self.n_heads, p=3)
        Z = self.matmul1(Q, K)
        Z = Z / (self.head_dim ** 0.5)
        Z = self.softmax(Z)
        Z = self.dropout(Z)
        Y = self.matmul2(Z, V)
        Y = rearrange(Y, 'b h n d -> b n (h d)')
        Y = self.out_proj(Y)
        return Y

    def relprop(
        self, 
        R : torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        """
        R = self.out_proj.relprop(R, **kwargs)
        R = rearrange(R, 'b n (h d) -> b h n d', h=self.n_heads)
        RZ, RV = self.matmul2.relprop(R, **kwargs)
        RZ = self.dropout.relprop(RZ, **kwargs)
        RZ = self.softmax.relprop(RZ, **kwargs)
        RQ, RK = self.matmul1.relprop(RZ, **kwargs)
        R_QKV = rearrange([RQ, RK, RV], 'p b h n d -> b n (p h d)', h=self.n_heads, p=3)
        R = self.qkv_nn.relprop(R_QKV, **kwargs)
        return R

class RelPropTransformerLayer(torch.nn.Module):
    r"""
    One layer of the Transformer encoder with support for Relevance Propagation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim : int = None,
        att_dim: int = 512,
        n_heads: int = 4,
        use_mlp: bool = True,
        dropout: float = 0.0
    ):
        """
        """
        super().__init__()

        self.att_module = RelPropMultiheadSelfAttention(
            att_dim=att_dim, 
            in_dim=in_dim, 
            out_dim=att_dim, 
            n_heads=n_heads,
            dropout=dropout
        )

        if out_dim is None:
            out_dim = in_dim
        
        if in_dim != att_dim:
            self.in_proj = RelPropLinear(in_dim, att_dim)
        else:
            self.in_proj = RelPropIdentity()
        
        self.use_mlp = use_mlp
        if use_mlp:
            self.mlp_module = RelPropSequential(
                RelPropLinear(att_dim, 4*att_dim),
                RelPropReLU(),
                RelPropDropout(dropout),
                RelPropLinear(4*att_dim, att_dim),
                RelPropDropout(dropout)
            )
        
        if out_dim != att_dim:
            self.out_proj = RelPropLinear(att_dim, out_dim)
        else:
            self.out_proj = RelPropIdentity()

        self.norm1 = RelPropLayerNorm(in_dim)
        self.norm2 = RelPropLayerNorm(att_dim)

        self.add1 = RelPropAdd2()
        self.add2 = RelPropAdd2()

        self.clone1 = RelPropClone()
        self.clone2 = RelPropClone()
    
    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        """
        X1, X2 = self.clone1(X, 2)
        Y = self.add1(self.in_proj(X1), self.att_module(self.norm1(X2)))
        if self.use_mlp:
            Y1, Y2 = self.clone2(Y, 2)
            Y = self.add2(Y1, self.mlp_module(self.norm2(Y2)))
        Y = self.out_proj(Y)
        return Y

    def relprop(
        self, 
        R : torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        """
        R = self.out_proj.relprop(R, **kwargs)
        if self.use_mlp:
            (R1, R2) = self.add2.relprop(R, **kwargs)
            R2 = self.mlp_module.relprop(R2, **kwargs)
            R2 = self.norm2.relprop(R2, **kwargs)
            R = self.clone2.relprop((R1, R2), **kwargs)
        
        (R1, R2) = self.add1.relprop(R, **kwargs)
        R1 = self.in_proj.relprop(R1, **kwargs)
        R2 = self.att_module.relprop(R2, **kwargs)
        R2 = self.norm1.relprop(R2, **kwargs)
        R = self.clone1.relprop((R1, R1), **kwargs)
        return R

class RelPropTransformerEncoder(torch.nn.Module):
    r"""

    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        att_dim: int = 512,
        n_heads: int = 4,
        n_layers: int = 4,
        use_mlp: bool = True,
        dropout: float = 0.0
    ):
        """
        Arguments:
            in_dim: Input dimension.
            out_dim: Output dimension. If None, out_dim = in_dim.
            att_dim: Attention dimension.
            n_heads: Number of heads.
            n_layers: Number of layers.
            use_mlp: Whether to use feedforward layer.
            add_self: Whether to add input to output.
            dropout: Dropout rate.        
        """

        super().__init__()

        if out_dim is None:
            out_dim = in_dim
        
        self.layers = torch.nn.ModuleList([
            RelPropTransformerLayer(
                in_dim=in_dim if i == 0 else att_dim, 
                out_dim=out_dim if i == n_layers - 1 else att_dim,
                att_dim=att_dim,  n_heads=n_heads, use_mlp=use_mlp, dropout=dropout
            ) 
            for i in range(n_layers)
        ])        
        self.norm = RelPropLayerNorm(out_dim)

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward method.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.

        Returns:
            Y: Output tensor of shape `(batch_size, bag_size, in_dim)`.        
        """

        Y = X
        for layer in self.layers:
            Y = layer(Y)
        
        Y = self.norm(Y) # (batch_size, bag_size, att_dim)

        return Y

    def relprop(
        self,
        R: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        """
        R = self.norm.relprop(R, **kwargs)
        for layer in self.layers[::-1]:
            R = layer.relprop(R, **kwargs)
        return R

                