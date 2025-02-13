from typing import Union

import torch

from torchmil.models.mil_model import MILModel

from torchmil.nn import SmAttentionPool, LazyLinear

from torchmil.nn.utils import get_feat_dim

class SmABMIL(MILModel):
    """
    Attention-based Multiple Instance Learning (ABMIL) model with the Sm operator.

    Proposed in the paper [Sm: enhanced localization in Multiple Instance Learning for medical imaging classification](https://arxiv.org/abs/2410.03276).
    """

    def __init__(
        self,
        in_shape: tuple = None,
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
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension). If not provided, it will be lazily initialized.
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

        self.feat_ext = feat_ext
        if in_shape is not None:
            feat_dim = get_feat_dim(feat_ext, in_shape)
        else:
            feat_dim = None
        self.pool = SmAttentionPool(
            in_dim=feat_dim,
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
        self.last_layer = LazyLinear(feat_dim, 1)

        self.criterion = criterion

    def forward(
        self,
        X: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
        return_att: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention values (before normalization) in addition to `Y_pred`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
        """

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)

        out_pool = self.pool(X, adj, mask, return_att)  # (batch_size, feat_dim)

        if return_att:
            Z, f = out_pool  # (batch_size, feat_dim), (batch_size, bag_size)
        else:
            Z = out_pool  # (batch_size, feat_dim)

        Y_pred = self.last_layer(Z)  # (batch_size, 1)
        Y_pred = Y_pred.squeeze(-1)  # (batch_size,)

        if return_att:
            return Y_pred, f
        else:
            return Y_pred

    def compute_loss(
        self,
        Y: torch.Tensor,
        X: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value.
        """

        Y_pred = self.forward(X, adj, mask, return_att=False)

        crit_loss = self.criterion(Y_pred.float(), Y.float())
        crit_name = self.criterion.__class__.__name__

        return Y_pred, {crit_name: crit_loss}

    def predict(
        self,
        X: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
        return_inst_pred: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_inst_pred (bool): If `True`, returns instance labels predictions, in addition to bag label predictions.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            y_inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        return self.forward(X, adj, mask, return_att=return_inst_pred)
