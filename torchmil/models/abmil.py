import torch
from torch import Tensor

from torchmil.models.modules import AttentionPool
# from torchmil.models import MILModel


class ABMIL(torch.nn.Module):
    """
    Attention-based Multiple Instance Learning (ABMIL) model. 

    Proposed in the paper [Attention-based Multiple Instance Learning](https://arxiv.org/abs/1802.04712).
    """

    def __init__(
        self,
        input_shape: tuple,
        att_dim: int = 128,
        att_act: str = 'tanh',
        feat_ext: torch.nn.Module = None,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        """
        Arguments:
            input_shape: Shape of input data expected by the feature extractor (excluding batch dimension).
            att_dim: Attention dimension.
            att_act: Activation function for attention. Possible values: 'tanh', 'relu', 'gelu'.
            feat_ext: Feature extractor.
            criterion: Loss function.
        """
        super().__init__()
        self.input_shape = input_shape
        self.criterion = criterion

        if feat_ext is None:
            self.feat_ext = torch.nn.Identity()
        else:
            self.feat_ext = feat_ext
        self.feat_dim = self._get_feat_dim()
        self.pool = AttentionPool(
            in_dim=self.feat_dim, att_dim=att_dim, act=att_act)
        self.classifier = torch.nn.Linear(self.feat_dim, 1)

    def _get_feat_dim(self) -> int:
        """
        Get feature dimension of the feature extractor.
        """
        with torch.no_grad():
            return self.feat_ext(torch.zeros((1, *self.input_shape))).shape[-1]

    def forward(
        self,
        X: Tensor,
        mask: Tensor,
        return_att: bool = False
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention values (before normalization) in addition to `Y_logits_pred`.

        Returns:
            Y_logits_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
        """

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)

        out_pool = self.pool(X, mask, return_att)  # (batch_size, feat_dim)

        if return_att:
            Z, f = out_pool  # (batch_size, feat_dim), (batch_size, bag_size)
        else:
            Z = out_pool  # (batch_size, feat_dim)

        Y_logits_pred = self.classifier(Z)  # (batch_size, 1)
        Y_logits_pred = Y_logits_pred.squeeze(-1)  # (batch_size,)

        if return_att:
            return Y_logits_pred, f
        else:
            return Y_logits_pred

    def compute_loss(
        self,
        Y_true: Tensor,
        X: Tensor,
        mask: Tensor
    ) -> tuple[Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            Y_true: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.

        Returns:
            Y_logits_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value.
        """

        Y_logits_pred = self.forward(X, mask, return_att=False)

        crit_loss = self.criterion(Y_logits_pred.float(), Y_true.float())
        crit_name = self.criterion.__class__.__name__

        return Y_logits_pred, {crit_name: crit_loss}

    @torch.no_grad()
    def predict(
        self,
        X: Tensor,
        mask: Tensor,
        return_y_pred: bool = True
    ) -> tuple[Tensor, Tensor]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_y_pred: If `True`, returns instance labels predictions, in addition to bag label predictions.

        Returns:
            T_logits_pred: Bag label logits of shape `(batch_size,)`.
            y_pred: If `return_y_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        return self.forward(X, mask, return_att=return_y_pred)
