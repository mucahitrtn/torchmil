import torch
from torch import Tensor

from .mil_model import MILModel
from torchmil.nn import ProbSmoothAttentionPool, get_feat_dim, LazyLinear

class ProbSmoothABMIL(MILModel):
    def __init__(
        self,
        in_shape: tuple = None,
        att_dim: int = 128,
        covar_mode: str = 'diag',
        n_samples_train: int = 1000,
        n_samples_test: int = 5000,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        """
        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension). If not provided, it will be lazily initialized.
            att_dim: Attention dimension.
            covar_mode: Covariance mode for the Gaussian prior. Possible values: 'diag', 'full'.
            n_samples_train: Number of samples for training.
            n_samples_test: Number of samples for testing.
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
            covar_mode=covar_mode,
            n_samples_train=n_samples_train,
            n_samples_test=n_samples_test
        )
        self.classifier = LazyLinear(feat_dim, 1)

    def forward(
        self,
        X: Tensor,
        adj: Tensor,
        mask: Tensor = None,
        return_att: bool = False,
        return_samples: bool = False,
        return_kl_div: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`. Only required when `return_kl_div=True`.
            return_att: If True, returns attention values (before normalization) in addition to `Y_pred`.
            return_samples: If True and `return_att=True`, the attention values returned are samples from the attention distribution.
            return_kl_div: If True, returns the KL divergence between the attention distribution and the prior distribution.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size, n_samples)` if `return_samples=True`, else `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape `(batch_size, bag_size, n_samples)` if `return_samples=True`, else `(batch_size, bag_size)`.
            kl_div: Only returned when `return_kl_div=True`. KL divergence between the attention distribution and the prior distribution of shape `()`.
        """

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)

        out_pool = self.pool(X, adj, mask, return_att=return_att, return_kl_div=return_kl_div)

        if return_kl_div:
            if return_att:
                z, f, kl_div = out_pool
            else:
                z, kl_div = out_pool
        else:
            if return_att:
                z, f = out_pool
            else:
                z = out_pool

        z = z.transpose(1, 2)  # (batch_size, n_samples, feat_dim)
        Y_pred = self.classifier(z)  # (batch_size, n_samples, 1)
        Y_pred = Y_pred.squeeze(-1)  # (batch_size, n_samples)

        if not return_samples:
            Y_pred = Y_pred.mean(dim=-1)  # (batch_size,)
            if return_att:
                f = f.mean(dim=-1)  # (batch_size, bag_size)

        if return_kl_div:
            if return_att:
                return Y_pred, f, kl_div
            else:
                return Y_pred, kl_div
        else:
            if return_att:
                return Y_pred, f
            else:
                return Y_pred

    def compute_loss(
        self,
        Y: Tensor,
        X: Tensor,
        adj: Tensor,
        mask: Tensor = None
    ) -> tuple[Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value and the KL divergence between the attention distribution and the prior distribution.
        """

        Y_pred, kl_div = self.forward(
            X, adj, mask, return_att=False, return_samples=True, return_kl_div=True)  # (batch_size, n_samples)
        Y_pred_mean = Y_pred.mean(dim=-1)  # (batch_size,)

        Y = Y.unsqueeze(-1).expand(-1, Y_pred.shape[-1])
        crit_loss = self.criterion(Y_pred.float(), Y.float())
        crit_name = self.criterion.__class__.__name__

        return Y_pred_mean, {crit_name: crit_loss, 'KLDiv': kl_div}

    def predict(self,
        X: Tensor,
        adj: Tensor,
        mask: Tensor = None,
        return_inst_pred: bool = True,
        return_samples: bool = False
    ) -> tuple[Tensor, Tensor]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_inst_pred: If True, returns the attention values as instance labels predictions, in addition to bag label predictions.
            return_samples: If True and `return_inst_pred=True`, the instance label predictions returned are samples from the instance label distribution.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            y_inst_pred: Only returned when `return_inst_pred=True`. Attention values (before normalization) of shape `(batch_size, bag_size)` if `return_samples=False`, else `(batch_size, bag_size, n_samples)`.
        """
        Y_pred, att_val = self.forward(X, adj, mask, return_att=return_inst_pred, return_samples=return_samples)
        return Y_pred, att_val

class SmoothABMIL(ProbSmoothABMIL):
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
        super(SmoothABMIL).__init__(
            in_shape=in_shape,
            att_dim=att_dim,
            covar_mode='zero',
            n_samples_train=1,
            n_samples_test=1,
            feat_ext=feat_ext,
            criterion=criterion
        )

    def compute_loss(
        self,
        Y: Tensor,
        X: Tensor,
        adj: Tensor,
        mask: Tensor = None
    ) -> tuple[Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size
            mask: Mask of shape `(batch_size, bag_size)`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value and the Dirichlet energy of the attention values.
        """

        Y_pred, loss_dict = super().compute_loss(Y, X, adj, mask)
        loss_dict['DirEnergy'] = loss_dict.pop('KLDiv')

        return Y_pred, loss_dict