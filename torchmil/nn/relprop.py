import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import warnings

from einops import rearrange


__all__ = ['forward_hook', 'Clone', 'Add', 'Cat', 'ReLU', 'GELU', 'Dropout', 'BatchNorm2d', 'Linear', 'MaxPool2d',
           'AdaptiveAvgPool2d', 'AvgPool2d', 'Conv2d', 'Sequential', 'safe_divide', 'einsum', 'Softmax', 'IndexSelect',
           'LayerNorm', 'AddEye']

def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True

    self.Y = output


def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha=0.5):
        return R

class RelPropSimple(RelProp):
    def relprop(self, R, alpha=0.5):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs

class AddEye(RelPropSimple):
    # input of shape B, C, seq_len, seq_len
    def forward(self, input):
        return input + torch.eye(input.shape[2]).expand_as(input).to(input.device)

class ReLU(nn.ReLU, RelProp):
    pass

class GELU(nn.GELU, RelProp):
    pass

class Softmax(nn.Softmax, RelProp):
    pass

class LayerNorm(nn.LayerNorm, RelProp):
    pass

class Dropout(nn.Dropout, RelProp):
    pass


class MaxPool2d(nn.MaxPool2d, RelPropSimple):
    pass

class LayerNorm(nn.LayerNorm, RelProp):
    pass

class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, RelPropSimple):
    pass


class AvgPool2d(nn.AvgPool2d, RelPropSimple):
    pass


class Add(RelPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha=0.5):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        a = self.X[0] * C[0]
        b = self.X[1] * C[1]

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())

        outputs = [a, b]

        return outputs

class Identity(nn.Identity, RelProp):
    def relprop(self, R, alpha=0.5):
        return R

class Einsum(RelPropSimple):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation
    def forward(self, *operands):
        return torch.einsum(self.equation, *operands)

class IndexSelect(RelProp):
    def forward(self, inputs, dim, indices):
        self.__setattr__('dim', dim)
        self.__setattr__('indices', indices)

        return torch.index_select(inputs, dim, indices)

    def relprop(self, R, alpha=0.5):
        Z = self.forward(self.X, self.dim, self.indices)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs

class Clone(RelProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def relprop(self, R, alpha=0.5):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.X, S)[0]

        R = self.X * C

        return R

class Cat(RelProp):
    def forward(self, inputs, dim):
        self.__setattr__('dim', dim)
        return torch.cat(inputs, dim)

    def relprop(self, R, alpha=0.5):
        Z = self.forward(self.X, self.dim)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        outputs = []
        for x, c in zip(self.X, C):
            outputs.append(x * c)

        return outputs

class Sequential(nn.Sequential):
    def relprop(self, R, alpha=0.5):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha)
        return R

class BatchNorm2d(nn.BatchNorm2d, RelProp):
    def relprop(self, R, alpha=0.5):
        X = self.X
        beta = 1 - alpha
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))
        Z = X * weight + 1e-9
        S = R / Z
        Ca = S * weight
        R = self.X * (Ca)
        return R

class Linear(nn.Linear, RelProp):
    def relprop(self, R, alpha=0.5):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            S1 = safe_divide(R, Z1 + Z2)
            S2 = safe_divide(R, Z1 + Z2)
            C1 = x1 * torch.autograd.grad(Z1, x1, S1)[0]
            C2 = x2 * torch.autograd.grad(Z2, x2, S2)[0]

            return C1 + C2

        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)

        R = alpha * activator_relevances - beta * inhibitor_relevances

        return R

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class MultiheadSelfAttention(RelProp):
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
            self.qkv_nn = Linear(in_dim, 3 * att_dim, bias=False)
        else:
            self.qkv_nn = Identity()

        if out_dim != att_dim:
            self.out_proj = Linear(att_dim, out_dim)
        else:
            self.out_proj = Identity()

        self.softmax = Softmax(dim=-1)
        self.dropout = Dropout(dropout)

        self.matmul1 = Einsum("b h i d, b h j d -> b h i j")
        self.matmul2 = Einsum("b h i j, b h j d -> b h i d")

        self.scale = 1.0 / (self.head_dim ** 0.5)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def save_att_grad(self, grad):
        self.att_grad = grad

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        """
        QKV = self.qkv_nn(X) # (batch_size, seq_len, 3 * att_dim)
        Q, K, V = rearrange(QKV, 'b n (p h d) -> p b h n d', h=self.n_heads, p=3)
        QK = self.matmul1([Q, K])*self.scale # (batch_size, n_heads, seq_len, seq_len)
        S = self.softmax(QK)
        S = self.dropout(S)
        Y = self.matmul2([S, V])
        Y = rearrange(Y, 'b h n d -> b n (h d)')
        Y = self.out_proj(Y)
        return Y

    def relprop(
        self,
        R : torch.Tensor,
        return_att_relevance: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        """
        R = self.out_proj.relprop(R, **kwargs)
        R = rearrange(R, 'b n (h d) -> b h n d', h=self.n_heads)
        R_S, RV = self.matmul2.relprop(R, **kwargs)
        R_S = self.dropout.relprop(R_S, **kwargs)
        R_QK = self.softmax.relprop(R_S, **kwargs)
        RQ, RK = self.matmul1.relprop(R_QK, **kwargs)
        R_QKV = rearrange([RQ, RK, RV], 'p b h n d -> b n (p h d)', h=self.n_heads, p=3)
        R = self.qkv_nn.relprop(R_QKV, **kwargs)
        if return_att_relevance:
            return R, R_S
        else:
            return R

class TransformerLayer(torch.nn.Module):
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

        self.att_module = MultiheadSelfAttention(
            att_dim=att_dim,
            in_dim=in_dim,
            out_dim=att_dim,
            n_heads=n_heads,
            dropout=dropout
        )

        if out_dim is None:
            out_dim = in_dim

        if in_dim != att_dim:
            self.in_proj = Linear(in_dim, att_dim)
        else:
            self.in_proj = Identity()

        self.use_mlp = use_mlp
        if use_mlp:
            self.mlp_module = Sequential(
                Linear(att_dim, 4*att_dim),
                ReLU(),
                Dropout(dropout),
                Linear(4*att_dim, att_dim),
                Dropout(dropout)
            )

        if out_dim != att_dim:
            self.out_proj = Linear(att_dim, out_dim)
        else:
            self.out_proj = Identity()

        self.norm1 = LayerNorm(in_dim)
        self.norm2 = LayerNorm(att_dim)

        self.add1 = Add()
        self.add2 = Add()

        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        """
        X1, X2 = self.clone1(X, 2)
        Y = self.add1([self.in_proj(X1), self.att_module(self.norm1(X2))])
        if self.use_mlp:
            Y1, Y2 = self.clone2(Y, 2)
            Y = self.add2([Y1, self.mlp_module(self.norm2(Y2))])
        Y = self.out_proj(Y)
        return Y

    def relprop(
        self,
        R : torch.Tensor,
        return_att_relevance: bool = False,
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
        if return_att_relevance:
            R2, R_S = self.att_module.relprop(R2, return_att_relevance=True, **kwargs)
        else:
            R2 = self.att_module.relprop(R2, **kwargs)
        R2 = self.norm1.relprop(R2, **kwargs)
        R = self.clone1.relprop((R1, R1), **kwargs)
        if return_att_relevance:
            return R, R_S
        else:
            return R

class TransformerEncoder(torch.nn.Module):
    r"""
    Transformer encoder with support for Relevance Propagation.
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
            TransformerLayer(
                in_dim=in_dim if i == 0 else att_dim,
                out_dim=out_dim if i == n_layers - 1 else att_dim,
                att_dim=att_dim,  n_heads=n_heads, use_mlp=use_mlp, dropout=dropout
            )
            for i in range(n_layers)
        ])
        self.norm = LayerNorm(out_dim)

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
        return_att_relevance: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        """
        R = self.norm.relprop(R, **kwargs)
        if return_att_relevance:
            att_rel_list = []
            for layer in self.layers[::-1]:
                R, R_S = layer.relprop(R, return_att_relevance=True, **kwargs)
                att_rel_list.append(R_S)
            return R, att_rel_list
        else:
            for layer in self.layers[::-1]:
                R = layer.relprop(R, **kwargs)
            return R
