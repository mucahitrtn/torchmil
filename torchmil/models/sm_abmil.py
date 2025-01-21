from typing import Union

import torch
from torch import Tensor

from torchmil.models.modules import SmAttentionPool


class SmABMIL(torch.nn.Module):
    """
    Attention-based Multiple Instance Learning (ABMIL) model with the Sm operator.

    Proposed in the paper [Sm: enhanced localization in Multiple Instance Learning for medical imaging classification](https://arxiv.org/abs/2410.03276).
    """

    def __init__(
        self,
        input_shape: tuple,
        att_dim: int = 128,
        att_act: str = 'tanh',
        sm_mode: str = 'approx',
        sm_alpha: Union[float, str] = 'trainable',
        sm_layers: int = 1,
        sm_steps: int = 10,
        sm_pre: bool = False,
        sm_post: bool = False,
        sm_spectral_norm: bool = False,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        """
        Arguments:
            input_shape (tuple): Shape of input data expected by the feature extractor (excluding batch dimension).
            att_dim (int): Attention dimension.
            att_act (str): Activation function for attention. Possible values: 'tanh', 'relu', 'gelu'.
            sm_mode (str): Mode for the Sm operator. Possible values: 'approx', 'exact'.
            sm_alpha (Union[float, str]): Alpha value for the Sm operator. If 'trainable', alpha is trainable.
            sm_layers (int): Number of layers that use the Sm operator.
            sm_steps (int): Number of steps for the Sm operator.
            sm_pre (bool): If True, apply Sm operator before the attention pooling.
            sm_post (bool): If True, apply Sm operator after the attention pooling.
            sm_spectral_norm (bool): If True, apply spectral normalization to all linear layers.
            feat_ext (torch.nn.Module): Feature extractor.
            criterion (torch.nn.Module): Loss function. By default, Binary Cross-Entropy loss from logits for binary classification.
        """
        super().__init__()
        self.input_shape = input_shape
        self.criterion = criterion

        self.feat_ext = feat_ext
        self.feat_dim = self._get_feat_dim()
        self.pool = SmAttentionPool(
            in_dim=self.feat_dim,
            att_dim=att_dim,
            act=att_act,
            sm_mode=sm_mode,
            sm_alpha=sm_alpha,
            sm_layers=sm_layers,
            sm_steps=sm_steps,
            sm_pre=sm_pre,
            sm_post=sm_post,
            sm_spectral_norm=sm_spectral_norm
        )
        self.last_layer = torch.nn.Linear(self.feat_dim, 1)

    def _get_feat_dim(self) -> int:
        """
        Get feature dimension of the feature extractor.
        """
        with torch.no_grad():
            return self.feat_ext(torch.zeros((1, *self.input_shape))).shape[-1]

    def forward(
        self,
        X: Tensor,
        adj_mat: Tensor,
        mask: Tensor,
        return_att: bool = False
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """
        Forward pass.

        Arguments:
            X (Tensor): Bag features of shape `(batch_size, bag_size, ...)`.
            adj_mat (Tensor): Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask (Tensor): Mask of shape `(batch_size, bag_size)`.
            return_att (bool): If True, returns attention values (before normalization) in addition to `bag_pred`.

        Returns:
            bag_pred (Tensor): Bag label logits of shape `(batch_size,)`.
            att (Tensor): Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size)..
        """

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)

        out_pool = self.pool(X, adj_mat, mask, return_att)  # (batch_size, feat_dim)

        if return_att:
            Z, f = out_pool  # (batch_size, feat_dim), (batch_size, bag_size)
        else:
            Z = out_pool  # (batch_size, feat_dim)

        bag_pred = self.last_layer(Z)  # (batch_size, 1)
        bag_pred = bag_pred.squeeze(-1)  # (batch_size,)

        if return_att:
            return bag_pred, f
        else:
            return bag_pred

    def compute_loss(
        self,
        Y_true: Tensor,
        X: Tensor,
        adj_mat: Tensor,
        mask: Tensor
    ) -> tuple[Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            Y_true (Tensor): Bag labels of shape `(batch_size,)`.
            X (Tensor): Bag features of shape `(batch_size, bag_size, ...)`.
            adj_mat (Tensor): Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask (Tensor): Mask of shape `(batch_size, bag_size)`.

        Returns:
            bag_pred (Tensor): Bag label logits of shape `(batch_size,)`.
            loss_dict (dict): Dictionary containing the loss value.
        """

        bag_pred = self.forward(X, adj_mat, mask, return_att=False)

        crit_loss = self.criterion(bag_pred.float(), Y_true.float())
        crit_name = self.criterion.__class__.__name__

        return bag_pred, {crit_name: crit_loss}

    @torch.no_grad()
    def predict(
        self,
        X: Tensor,
        adj_mat: Tensor,
        mask: Tensor,
        return_inst_pred: bool = True
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X (Tensor): Bag features of shape `(batch_size, bag_size, ...)`.
            adj_mat (Tensor): Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask (Tensor): Mask of shape `(batch_size, bag_size)`.
            return_inst_pred (bool): If `True`, returns instance labels predictions, in addition to bag label predictions.

        Returns:
            T_logits_pred (Tensor): Bag label logits of shape `(batch_size,)`.
            inst_pred (Tensor): If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.

        """
        return self.forward(X, adj_mat, mask, return_att=return_inst_pred)
