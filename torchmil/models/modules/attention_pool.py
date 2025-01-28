from typing import Union

import torch
from torch import Tensor

from torchmil.models.modules.utils import masked_softmax
from .utils import LazyLinear


class AttentionPool(torch.nn.Module):
    """
    Multiple Instance Learning (MIL) attention pooling layer. 

    Proposed in in the paper [Attention-based Multiple Instance Learning](https://arxiv.org/abs/1802.04712).
    """

    def __init__(
        self, 
        in_dim : int = None,
        att_dim : int = 128,
        act : str = 'tanh',
        gated : bool = False
    ) -> None:
        """
        Arguments:
            in_dim: Input dimension. If not provided, it will be lazily initialized.
            att_dim: Attention dimension.
            act: Activation function for attention. Possible values: 'tanh', 'relu', 'gelu'.
            gated: If True, use gated attention.
        """

        super(AttentionPool, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.act = act
        self.gated = gated

        self.fc1 = LazyLinear(in_dim, att_dim)
        self.fc2 = torch.nn.Linear(att_dim, 1, bias=False)

        if self.gated:
            self.fc_gated = LazyLinear(in_dim, att_dim)
            self.act_gated = torch.nn.Sigmoid()

        if self.act == 'tanh':
            self.act_layer = torch.nn.Tanh()
        elif self.act == 'relu':
            self.act_layer = torch.nn.ReLU()
        elif self.act == 'gelu':
            self.act_layer = torch.nn.GELU()
        else:
            raise ValueError(f"[{self.__class__.__name__}] act must be 'tanh', 'relu' or 'gelu'")   
    
    def forward(
        self, 
        X : Tensor,
        mask : Tensor = None,
        return_att : bool = False
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, in_dim)`.
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

        H = self.fc1(X) # (batch_size, bag_size, att_dim)
        H = self.act_layer(H) # (batch_size, bag_size, att_dim)

        if self.gated:
            G = self.fc_gated(X)
            G = self.act_gated(G)
            H = H * G

        f = self.fc2(H) # (batch_size, bag_size, 1)

        s = masked_softmax(f, mask, dim=1)
        z = torch.bmm(X.transpose(1,2), s).squeeze(dim=-1) # (batch_size, D)

        if return_att:
            return z, f.squeeze(dim=-1)
        else:
            return z