from typing import Union

import torch
from torch import Tensor

from torchmil.models.modules.utils import masked_softmax
from torchmil.models.modules.Sm import ApproxSm, ExactSm


class AttentionPool(torch.nn.Module):
    """
    Multiple Instance Learning (MIL) attention pooling layer. 

    Proposed in in the paper [Attention-based Multiple Instance Learning](https://arxiv.org/abs/1802.04712).
    """

    def __init__(
        self, 
        in_dim : int,
        att_dim : int = 128,
        act : str = 'tanh',
        gated : bool = False
    ) -> None:
        """
        Arguments:
            in_dim: Input dimension.
            att_dim: Attention dimension.
            act: Activation function for attention. Possible values: 'tanh', 'relu', 'gelu'.
            gated: If True, use gated attention.
        """

        super(AttentionPool, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.act = act
        self.gated = gated

        self.fc1 = torch.nn.Linear(in_dim, att_dim)
        self.fc2 = torch.nn.Linear(att_dim, 1, bias=False)

        if self.gated:
            self.fc_gated = torch.nn.Linear(in_dim, att_dim)
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

class SmAttentionPool(torch.nn.Module):
    """
    Multiple Instance Learning (MIL) attention pooling layer with the Sm operator.

    Proposed in the paper [Sm: enhanced localization in Multiple Instance Learning for medical imaging classification](https://arxiv.org/abs/2410.03276).
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
        **Arguments**:
        
        - in_dim (int): Input dimension.
        - att_dim (int): Attention dimension.
        - act (str): Activation function for attention. Possible values: 'tanh', 'relu', 'gelu'.
        - sm_mode (str): Mode for the Sm operator. Possible values: 'approx', 'exact'.
        - sm_alpha (Union[float, str]): Alpha value for the Sm operator. If 'trainable', alpha is trainable.
        - sm_layers (int): Number of layers that use the Sm operator.
        - sm_steps (int): Number of steps for the Sm operator.
        - sm_pre (bool): If True, apply Sm operator before the attention pooling.
        - sm_post (bool): If True, apply Sm operator after the attention pooling.
        - sm_spectral_norm (bool): If True, apply spectral normalization to all linear layers.
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

        if self.sm_mode == 'approx':
            sm_fn = lambda : ApproxSm(alpha = sm_alpha, num_steps = sm_steps)
        elif self.sm_mode == 'exact':
            sm_fn = lambda : ExactSm(alpha = sm_alpha)        
        
        self.sm = sm_fn()

        self.mlp = torch.nn.ModuleList()
        for _ in range(sm_layers):
            self.mlp.append(torch.nn.Linear(in_dim, in_dim))
        
        if self.sm_spectral_norm:
            self.apply(self._init_spectral_norm)
            
    def _init_spectral_norm(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.utils.spectral_norm(m)
    
    def forward(
            self, 
            X : Tensor,
            adj_mat : Tensor,
            mask : Tensor = None,
            return_att : bool = False
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Arguments:
            X (Tensor): Bag features of shape `(batch_size, bag_size, in_dim)`.
            adj_mat (Tensor): Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask (Tensor): Mask of shape `(batch_size, bag_size)`.
            return_att (bool): If True, returns attention values (before normalization) in addition to `z`.
        
        Returns:
            z (Tensor): Bag representation of shape `(batch_size, in_dim)`.
            f (Tensor): Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
        """

        batch_size = X.shape[0]
        bag_size = X.shape[1]
        
        if mask is None:
            mask = torch.ones(batch_size, bag_size, device=X.device)
        mask = mask.unsqueeze(dim=-1) # (batch_size, bag_size, 1)

        if self.sm_pre:
            X = self.sm(X, adj_mat) # (batch_size, bag_size, in_dim)

        H = self.proj1(X) # (batch_size, bag_size, att_dim)
        H = self.act_layer(H) # (batch_size, bag_size, att_dim)

        for layer in self.mlp:
            H = layer(H)
            H = self.sm(H, adj_mat)
            H = self.act_layer(H)

        f = self.proj2(H) # (batch_size, bag_size, 1)

        s = masked_softmax(f, mask, dim=1) # (batch_size, bag_size, 1)
        z = torch.bmm(X.transpose(1,2), s).squeeze(dim=-1) # (batch_size, D)

        if return_att:
            return z, f.squeeze(dim=-1)
        else:
            return z