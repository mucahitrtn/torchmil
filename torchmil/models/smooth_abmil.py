import torch
from torch import Tensor

from torchmil.models.modules import ProbSmoothAttentionPool

from torchmil.models.modules.utils import get_feat_dim, LazyLinear


class SmoothABMIL(torch.nn.Module):
    def __init__(
        self,
        in_shape: tuple = None,
        att_dim: int = 128,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        """
        Class constructor.

        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension). If not provided, it will be lazily initialized.
            att_dim: Attention dimension.
            feat_ext: Feature extractor.
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits for binary classification.
        """
        super().__init__()
        self.criterion = criterion

        self.feat_ext = feat_ext
        if in_shape is not None:
            feat_dim = get_feat_dim(feat_ext, in_shape)
        else:
            feat_dim = None
        self.pool = ProbSmoothAttentionPool(
            in_dim=feat_dim,
            att_dim=att_dim,
            covar_mode='zero',
            n_samples_train=0,
            n_samples_test=0
        )
        self.classifier = LazyLinear(feat_dim, 1)


    def forward(
        self,
        X: Tensor,
        mask: Tensor = None,
        adj_mat: Tensor = None,
        return_att: bool = False,
        return_samples: bool = False,
        return_dir_energy: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            adj_mat: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`. Only required when `return_kl_div=True`.
            return_att: If True, returns attention values (before normalization) in addition to `bag_pred`.
            return_samples: If True and `return_att=True`, the attention values returned are samples from the attention distribution.
            return_dir_energy: If True, returns the Dirichlet energy of the attention values.

        Returns:
            bag_pred: Bag label logits of shape `(batch_size, n_samples)` if `return_samples=True`, else `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape `(batch_size, bag_size, n_samples)` if `return_samples=True`, else `(batch_size, bag_size)`.
            dir_energy: Only returned when `return_dir_energy=True`. Dirichlet energy of the attention values of shape `(batch_size,)`.
        """

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)

        out_pool = self.pool(
            X, mask, adj_mat, return_att=return_att, return_kl_div=return_dir_energy)

        if return_dir_energy:
            if return_att:
                Z, f, dir_energy = out_pool
            else:
                Z, dir_energy = out_pool
        else:
            if return_att:
                Z, f = out_pool
            else:
                Z = out_pool

        Z = Z.transpose(1, 2)  # (batch_size, n_samples, feat_dim)
        bag_pred = self.classifier(Z)  # (batch_size, n_samples, 1)
        bag_pred = bag_pred.squeeze(-1)  # (batch_size, n_samples)

        if not return_samples:
            bag_pred = bag_pred.mean(dim=-1)  # (batch_size,)
            if return_att:
                f = f.mean(dim=-1)  # (batch_size, bag_size)

        if return_dir_energy:
            if return_att:
                return bag_pred, f, dir_energy
            else:
                return bag_pred, dir_energy
        else:
            if return_att:
                return bag_pred, f
            else:
                return bag_pred

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
            bag_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value and the Dirichlet energy of the attention values.
        """

        bag_pred, dir_energy = self.forward(
            X, mask, adj_mat, return_att=False, return_samples=True, return_kl_div=True)  # (batch_size, n_samples)
        bag_pred_mean = bag_pred.mean(dim=-1)  # (batch_size,)

        Y_true = Y_true.unsqueeze(-1).expand(-1, bag_pred.shape[-1])
        crit_loss = self.criterion(bag_pred.float(), Y_true.float())
        crit_name = self.criterion.__class__.__name__

        return bag_pred_mean, {crit_name: crit_loss, 'DirEnergy': dir_energy}

    @torch.no_grad()
    def predict(self,
        X: Tensor,
        mask: Tensor,
        return_inst_pred: bool = True,
        return_samples: bool = False
    ) -> tuple[Tensor, Tensor]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_inst_pred: If True, returns the attention values as instance labels predictions, in addition to bag label predictions.
            return_samples: If True and `return_inst_pred=True`, the instance label predictions returned are samples from the instance label distribution.

        Returns:
            bag_pred: Bag label logits of shape `(batch_size,)`.
            att_val: Only returned when `return_inst_pred=True`. Attention values (before normalization) of shape `(batch_size, bag_size)` if `return_samples=False`, else `(batch_size, bag_size, n_samples)`.

        """
        bag_pred, att_val = self.forward(
            X, mask, return_att=return_inst_pred, return_samples=return_samples)
        return bag_pred, att_val
