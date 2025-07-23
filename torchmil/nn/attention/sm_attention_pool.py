from typing import Union

import torch
from torch import Tensor

from torchmil.nn.utils import masked_softmax
from torchmil.nn.sm import Sm


class SmAttentionPool(torch.nn.Module):
    r"""
    Attention-based pooling with the Sm operator, as proposed in [Sm: enhanced localization in Multiple Instance Learning for medical imaging classification](https://arxiv.org/abs/2410.03276).

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times \texttt{in_dim}}$,
    this module aggregates the instance features into a bag representation in a similar way as [ABMIL](../../models/abmil.md), 
    but it incorporates the $\texttt{Sm}$ operator to promote smoothness in the attention values, see [Sm](../sm.md) for more details.

    Formally, if $\texttt{sm_where == "early"}$, the module first applies the $\texttt{Sm}$ operator to the input bag features $\mathbf{X}$,
    
    \begin{gather}
        \mathbf{X} = \operatorname{Sm}(\mathbf{X}).
    \end{gather}

    Then, it computes the attention values $\mathbf{f} \in \mathbb{R}^{N}$ as,

    \begin{gather}
        \mathbf{f} = \operatorname{act}( \operatorname{Sm}(\mathbf{X} \mathbf{W}^\top)) \mathbf{w}, \quad \text{if } \texttt{sm_where == "mid"},\\
        \mathbf{f} = \operatorname{Sm}(\operatorname{act}(\mathbf{X} \mathbf{W}^\top) \mathbf{w}), \quad \text{if } \texttt{sm_where == "late"}, 
    \end{gather}

    where $\mathbf{W} \in \mathbb{R}^{\texttt{in_dim} \times \texttt{att_dim}}$ and $\mathbf{w} \in \mathbb{R}^{\texttt{att_dim} \times 1}$ are learnable parameters, and $\operatorname{act}$ is the activation function.
    Then, it computes the bag representation $\mathbf{z} \in \mathbb{R}^{\texttt{in_dim}}$ as,

    \begin{gather}
        \mathbf{z} = \mathbf{X}^\top \operatorname{Softmax}(\mathbf{f}) = \sum_{n=1}^N s_n \mathbf{x}_n,
    \end{gather}
    
    where $s_n$ is the normalized attention score for the $n$-th instance.

    **Spectral normalization:** To ensure that the Dirichlet energy decreases after applying the $\texttt{Sm}$ operator, the linear layers can be optionally normalized using spectral normalization.
    In the original paper, this results in better performance.
    If `spectral_norm=True`, the linear layers after the $\texttt{Sm}$ operator are normalized using spectral normalization.
    """

    def __init__(
        self,
        in_dim: int,
        att_dim: int = 128,
        act: str = "gelu",
        sm_mode: str = "approx",
        sm_alpha: Union[float, str] = "trainable",
        sm_steps: int = 10,
        sm_where: str = "early",
        spectral_norm: bool = False,
    ):
        """
        Arguments:
            in_dim: Input dimension.
            att_dim: Attention dimension.
            act: Activation function for attention. Possible values: 'tanh', 'relu', 'gelu'.
            sm_mode: Mode for the Sm operator. Possible values: 'approx', 'exact'.
            sm_alpha: Alpha value for the Sm operator. If 'trainable', alpha is trainable.
            sm_steps: Number of steps for the Sm operator.
            sm_where: Where to apply the Sm operator. Possible values: 'early', 'mid', 'late', 'none'.
            spectral_norm: If True, apply spectral normalization to linear layers. If `sm_where` is 'none', all linear layers are normalized.
        """

        super(SmAttentionPool, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.act = act
        self.sm_mode = sm_mode
        self.sm_alpha = sm_alpha
        self.sm_steps = sm_steps
        self.sm_where = sm_where
        self.spectral_norm = spectral_norm

        if self.act == "tanh":
            self.act_layer_fn = torch.nn.functional.tanh
        elif self.act == "relu":
            self.act_layer_fn = torch.nn.functional.relu
        elif self.act == "gelu":
            self.act_layer_fn = torch.nn.functional.gelu
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] act must be 'tanh', 'relu' or 'gelu'"
            )

        if (sm_steps > 0) and (sm_alpha not in [0.0, 0]) and sm_where != "none":
            self.sm = Sm(
                alpha=sm_alpha,
                num_steps=sm_steps,
                mode=sm_mode,
            )
        else:
            self.sm = torch.nn.Identity()

        self.proj1 = torch.nn.Linear(in_dim, att_dim)
        self.proj2 = torch.nn.Linear(att_dim, 1, bias=False)

        if self.spectral_norm:
            if self.sm_where == "mid":
                self.proj2 = torch.nn.utils.parametrizations.spectral_norm(self.proj2)
            elif self.sm_where in ["early", "none"]:
                self.proj1 = torch.nn.utils.parametrizations.spectral_norm(self.proj1)
                self.proj2 = torch.nn.utils.parametrizations.spectral_norm(self.proj2)

    def forward(
        self, X: Tensor, adj: Tensor, mask: Tensor = None, return_att: bool = False
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
        mask = mask.unsqueeze(dim=-1)  # (batch_size, bag_size, 1)

        if self.sm_where == "early":
            X = self.sm(X, adj)  # (batch_size, bag_size, in_dim)

        H = self.proj1(X)  # (batch_size, bag_size, att_dim)

        if self.sm_where == "mid":
            H = self.sm(H, adj)  # (batch_size, bag_size, att_dim)
        H = self.act_layer_fn(H)  # (batch_size, bag_size, att_dim)

        f = self.proj2(H)  # (batch_size, bag_size, 1)
        if self.sm_where == "late":
            f = self.sm(f, adj)

        s = masked_softmax(f, mask)  # (batch_size, bag_size, 1)
        z = torch.bmm(X.transpose(1, 2), s).squeeze(dim=-1)  # (batch_size, D)

        if return_att:
            return z, f.squeeze(dim=-1)
        else:
            return z
