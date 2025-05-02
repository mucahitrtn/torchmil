from typing import Union

import torch
from torch import Tensor

from torchmil.nn.utils import masked_softmax, LazyLinear
from torchmil.nn.sm import Sm

class SmAttentionPool(torch.nn.Module):
    r"""
    Attention-based pooling with the Sm operator, as proposed in [Sm: enhanced localization in Multiple Instance Learning for medical imaging classification](https://arxiv.org/abs/2410.03276).

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times \texttt{in_dim}}$,
    this model aggregates the instance features into a bag representation $\mathbf{z} \in \mathbb{R}^{\texttt{in_dim}}$ as,

    \begin{gather}
        \mathbf{f} = \operatorname{SmMLP}(\mathbf{X}) \in \mathbb{R}^{N}, \\
        \mathbf{z} = \mathbf{X}^\top \operatorname{Softmax}(\mathbf{f}) = \sum_{n=1}^N s_n \mathbf{x}_n,
    \end{gather}

    where $s_n$ is the normalized attention score for the $n$-th instance.

    To compute the attention values, $\operatorname{SmMLP}$ is defined as $\operatorname{SmMLP}(\mathbf{X}) = \mathbf{Y}^L$ where

    \begin{gather}
        \mathbf{Y}^0 = \mathbf{X}\mathbf{W^0}, \\
        \mathbf{Y}^l = \operatorname{act}( \texttt{Sm}(\mathbf{Y}^{l-1}\mathbf{W}^l)), \quad \text{for } l = 1, \ldots, L-1, \\
        \mathbf{Y}^L = \mathbf{Y}^{L-1}\mathbf{w},
    \end{gather}

    where $\mathbf{W^0} \in \mathbb{R}^{\texttt{in_dim} \times \texttt{att_dim}}$, $\mathbf{W}^l \in \mathbb{R}^{\texttt{att_dim} \times \texttt{att_dim}}$, $\mathbf{w} \in \mathbb{R}^{\texttt{att_dim} \times 1}$,
    $\operatorname{act} \ \colon \mathbb{R} \to \mathbb{R}$ is the activation function,
    and $\texttt{Sm}$ is the Sm operator, see [Sm](../sm.md) for more details.

    **Note**: If `sm_pre=True`, the Sm operator is applied before $\operatorname{SmMLP}$. If `sm_post=True`, the Sm operator is applied after $\operatorname{SmMLP}$.
    """

    def __init__(
            self,
            in_dim : int,
            att_dim : int = 128,
            act : str = 'gelu',
            sm_mode : str = 'approx',
            sm_alpha : Union[float, str] = 'trainable',
            sm_layers : int = 1,
            sm_steps : int = 10,
            sm_pre : bool = False,
            sm_post : bool = False,
            sm_spectral_norm : bool = False
        ):
        """
        Arguments:
            in_dim: Input dimension.
            att_dim: Attention dimension.
            act: Activation function for attention. Possible values: 'tanh', 'relu', 'gelu'.
            sm_mode: Mode for the Sm operator. Possible values: 'approx', 'exact'.
            sm_alpha: Alpha value for the Sm operator. If 'trainable', alpha is trainable.
            sm_layers: Number of layers that use the Sm operator.
            sm_steps: Number of steps for the Sm operator.
            sm_pre: If True, apply Sm operator before the attention pooling.
            sm_post: If True, apply Sm operator after the attention pooling.
            sm_spectral_norm: If True, apply spectral normalization to all linear layers.
        """

        super(SmAttentionPool, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.act = act
        self.sm_mode = sm_mode
        self.sm_alpha = sm_alpha
        self.sm_layers = sm_layers
        self.sm_steps = sm_steps
        self.sm_pre = sm_pre
        self.sm_post = sm_post
        self.sm_spectral_norm = sm_spectral_norm

        self.proj1 = torch.nn.Linear(in_dim, att_dim)
        self.proj2 = torch.nn.Linear(att_dim, 1, bias=False)

        if self.act == 'tanh':
            act_layer_fn = torch.nn.Tanh
        elif self.act == 'relu':
            act_layer_fn = torch.nn.ReLU
        elif self.act == 'gelu':
            act_layer_fn = torch.nn.GELU
        else:
            raise ValueError(f"[{self.__class__.__name__}] act must be 'tanh', 'relu' or 'gelu'")
        self.act_layer = act_layer_fn()

        self.sm = Sm(alpha = sm_alpha, num_steps = sm_steps, mode = sm_mode)

        self.mlp = torch.nn.ModuleList()
        for _ in range(sm_layers):
            linear = torch.nn.utils.parametrizations.spectral_norm(
                torch.nn.Linear(att_dim, att_dim)
            )
            self.mlp.append(linear)

        if self.sm_pre:
            self.proj1 = torch.nn.utils.parametrizations.spectral_norm(self.proj1)
            self.proj2 = torch.nn.utils.parametrizations.spectral_norm(self.proj2)

    def forward(
            self,
            X : Tensor,
            adj : Tensor,
            mask : Tensor = None,
            return_att : bool = False
        ) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, in_dim)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention values (before normalization) in addition to `z`.

        Returns:
            z: Bag representation of shape `(batch_size, in_dim)`.
            f: Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
        """

        batch_size = X.shape[0]
        bag_size = X.shape[1]

        if mask is None:
            mask = torch.ones(batch_size, bag_size, device=X.device)
        mask = mask.unsqueeze(dim=-1) # (batch_size, bag_size, 1)

        if self.sm_pre:
            X = self.sm(X, adj) # (batch_size, bag_size, in_dim)

        H = self.proj1(X) # (batch_size, bag_size, att_dim)
        H = self.act_layer(H) # (batch_size, bag_size, att_dim)

        for layer in self.mlp:
            H = self.sm(H, adj)
            H = layer(H)
            H = self.act_layer(H)

        f = self.proj2(H) # (batch_size, bag_size, 1)

        if self.sm_post:
            f = self.sm(f, adj)

        s = masked_softmax(f, mask) # (batch_size, bag_size, 1)
        z = torch.bmm(X.transpose(1,2), s).squeeze(dim=-1) # (batch_size, D)

        if return_att:
            return z, f.squeeze(dim=-1)
        else:
            return z
