import torch
from torch import Tensor

from torchmil.models.modules import AttentionPool, LazyLinear

from torchmil.models.modules.utils import get_feat_dim


class ABMIL(torch.nn.Module):
    r"""
    Attention-based Multiple Instance Learning (ABMIL) model, proposed in the paper [Attention-based Multiple Instance Learning](https://arxiv.org/abs/1802.04712).

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times D}$, this model aggregates the instance features into a bag representation $\mathbf{z} \in \mathbb{R}^{D}$ using the attention-based pooling, 

    $$ \mathbf{z} = \mathbf{X}^\top \text{Softmax}(\mathbf{f}) = \sum_{n=1}^N s_n \mathbf{x}_n, $$

    where $\mathbf{f} = \operatorname{MLP}(\mathbf{X}) \in \mathbb{R}^{N}$ are the attention values and $s_n$ is the normalized attention score for the $n$-th instance.
    The bag representation $\mathbf{z}$ is then fed into a classifier to predict the bag label.

    The model uses a two-layer perceptron for the attention values computation, and a linear layer for the bag label prediction.
    Optionally, the input bag may be passed through a feature extractor before the attention pooling.
    """

    def __init__(
        self,
        in_shape: tuple = None,
        att_dim: int = 128,
        att_act: str = 'tanh',
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        """
        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension). If not provided, it will be lazily initialized.
            att_dim: Attention dimension.
            att_act: Activation function for attention. Possible values: 'tanh', 'relu', 'gelu'.
            feat_ext: Feature extractor.
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits.
        """
        super().__init__()
        self.criterion = criterion

        self.feat_ext = feat_ext
        if in_shape is not None:
            feat_dim = get_feat_dim(feat_ext, in_shape)
        else:
            feat_dim = None
        self.pool = AttentionPool(
            in_dim=feat_dim, att_dim=att_dim, act=att_act)
        
        self.classifier = LazyLinear(in_features=feat_dim, out_features=1)

    def forward(
        self,
        X: Tensor,
        mask: Tensor = None,
        return_att: bool = False
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention values (before normalization) in addition to `bag_pred`.

        Returns:
            bag_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
        """

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)

        out_pool = self.pool(X, mask, return_att)  # (batch_size, feat_dim)

        if return_att:
            Z, f = out_pool  # (batch_size, feat_dim), (batch_size, bag_size)
        else:
            Z = out_pool  # (batch_size, feat_dim)

        bag_pred = self.classifier(Z)  # (batch_size, 1)
        bag_pred = bag_pred.squeeze(-1)  # (batch_size,)

        if return_att:
            return bag_pred, f
        else:
            return bag_pred

    def compute_loss(
        self,
        labels: Tensor,
        X: Tensor,
        mask: Tensor
    ) -> tuple[Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            labels: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.

        Returns:
            bag_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value.
        """

        bag_pred = self.forward(X, mask, return_att=False)

        crit_loss = self.criterion(bag_pred.float(), labels.float())
        crit_name = self.criterion.__class__.__name__

        return bag_pred, {crit_name: crit_loss}

    @torch.no_grad()
    def predict(
        self,
        X: Tensor,
        mask: Tensor,
        return_inst_pred: bool = True
    ) -> tuple[Tensor, Tensor]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_inst_pred: If `True`, returns instance labels predictions, in addition to bag label predictions.

        Returns:
            bag_pred: Bag label logits of shape `(batch_size,)`.
            inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        return self.forward(X, mask, return_att=return_inst_pred)
