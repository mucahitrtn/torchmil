import torch
from torch import Tensor

from torchmil.models.modules import ProbSmoothAttentionPool


class ProbSmoothABMIL(torch.nn.Module):
    def __init__(
        self,
        input_shape: tuple,
        att_dim: int = 128,
        covar_mode: str = 'diag',
        n_samples_train: int = 1000,
        n_samples_test: int = 5000,
        feat_ext: torch.nn.Module = None,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.criterion = criterion

        if feat_ext is None:
            self.feat_ext = torch.nn.Identity()
        else:
            self.feat_ext = feat_ext

        feat_dim = self._get_feat_dim()
        self.pool = ProbSmoothAttentionPool(
            in_dim=feat_dim,
            att_dim=att_dim,
            covar_mode=covar_mode,
            n_samples_train=n_samples_train,
            n_samples_test=n_samples_test
        )
        self.classifier = torch.nn.Linear(feat_dim, 1)

    def _get_feat_dim(self) -> int:
        """
        Get feature dimension of the feature extractor.
        """
        with torch.no_grad():
            return self.feat_ext(torch.zeros((1, *self.input_shape))).shape[-1]

    def forward(
        self,
        X: Tensor,
        mask: Tensor = None,
        adj_mat: Tensor = None,
        return_att: bool = False,
        return_samples: bool = False,
        return_kl_div: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            adj_mat: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`. Only required when `return_kl_div=True`.
            return_att: If True, returns attention values (before normalization) in addition to `Y_logits_pred`.
            return_samples: If True and `return_att=True`, the attention values returned are samples from the attention distribution.
            return_kl_div: If True, returns the KL divergence between the attention distribution and the prior distribution.

        Returns:
            Y_logits_pred: Bag label logits of shape `(batch_size, n_samples)` if `return_samples=True`, else `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape `(batch_size, bag_size, n_samples)` if `return_samples=True`, else `(batch_size, bag_size)`.
            kl_div: Only returned when `return_kl_div=True`. KL divergence between the attention distribution and the prior distribution of shape `()`.
        """

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)

        out_pool = self.pool(
            X, mask, adj_mat, return_att=return_att, return_kl_div=return_kl_div)

        if return_kl_div:
            if return_att:
                Z, f, kl_div = out_pool
            else:
                Z, kl_div = out_pool
        else:
            if return_att:
                Z, f = out_pool
            else:
                Z = out_pool

        Z = Z.transpose(1, 2)  # (batch_size, n_samples, feat_dim)
        Y_logits_pred = self.classifier(Z)  # (batch_size, n_samples, 1)
        Y_logits_pred = Y_logits_pred.squeeze(-1)  # (batch_size, n_samples)

        if not return_samples:
            Y_logits_pred = Y_logits_pred.mean(dim=-1)  # (batch_size,)
            if return_att:
                f = f.mean(dim=-1)  # (batch_size, bag_size)

        if return_kl_div:
            if return_att:
                return Y_logits_pred, f, kl_div
            else:
                return Y_logits_pred, kl_div
        else:
            if return_att:
                return Y_logits_pred, f
            else:
                return Y_logits_pred

    def compute_loss(
        self,
        Y_true: Tensor,
        X: Tensor,
        adj_mat: Tensor,
        mask: Tensor,
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

        Y_logits_pred, kl_div = self.forward(
            X, mask, adj_mat, return_att=False, return_samples=True, return_kl_div=True)  # (batch_size, n_samples)
        Y_logits_pred_mean = Y_logits_pred.mean(dim=-1)  # (batch_size,)

        Y_true = Y_true.unsqueeze(-1).expand(-1, Y_logits_pred.shape[-1])
        crit_loss = self.criterion(Y_logits_pred.float(), Y_true.float())
        crit_name = self.criterion.__class__.__name__

        return Y_logits_pred_mean, {crit_name: crit_loss, 'KLDiv': kl_div}

    @torch.no_grad()
    def predict(self,
        X: Tensor,
        mask: Tensor,
        return_y_pred: bool = True,
        return_samples: bool = False
    ) -> tuple[Tensor, Tensor]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_y_pred: If True, returns the attention values as instance labels predictions, in addition to bag label predictions.
            return_samples: If True and `return_y_pred=True`, the instance label predictions returned are samples from the instance label distribution.

        Returns:
            Y_logits_pred: Bag label logits of shape `(batch_size,)`.
            att_val: Only returned when `return_y_pred=True`. Attention values (before normalization) of shape `(batch_size, bag_size)` if `return_samples=False`, else `(batch_size, bag_size, n_samples)`.

        """
        Y_logits_pred, att_val = self.forward(
            X, mask, return_att=return_y_pred, return_samples=return_samples)
        return Y_logits_pred, att_val
