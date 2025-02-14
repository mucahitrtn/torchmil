import torch

from .mil_model import MILModel

from torchmil.nn import AttentionPool, TransformerEncoder
from torchmil.nn.utils import get_feat_dim

class GTP(MILModel):
    r"""
    """
    def __init__(
        self,
        in_shape : tuple,
        att_dim : int = 512,
        n_clusters : int = 100,
        n_layers : int = 1,
        n_heads : int = 8,
        use_mlp : bool = True,
        add_self : bool = True,
        dropout : float = 0.0,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        ) -> None:
        """
        Class constructor.

        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension). If not provided, it will be lazily initialized.
            pool_att_dim: Attention dimension for pooling.
            pool_act: Activation function for pooling. Possible values: 'tanh', 'relu', 'gelu'.
            pool_gated: If True, use gated attention in the attention pooling.
            feat_ext: Feature extractor.
            att_dim: Attention dimension for transformer encoder.
            n_layers: Number of layers in transformer encoder.
            n_heads: Number of heads in transformer encoder.
            use_mlp: Whether to use MLP in transformer encoder.
            add_self: Whether to add input to output in transformer encoder.
            dropout: Dropout rate in transformer encoder.
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits for binary classification.
        """
        super().__init__()
        self.criterion = criterion

        self.feat_ext = feat_ext
        feat_dim = get_feat_dim(feat_ext, in_shape)

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, feat_dim), requires_grad=True)
        self.transformer_encoder = TransformerEncoder(
            in_dim=feat_dim, 
            att_dim=att_dim, 
            n_layers=n_layers, 
            n_heads=n_heads, 
            use_mlp=use_mlp,
            add_self=add_self, 
            dropout=dropout
        )
        self.classifier = torch.nn.Linear(feat_dim, 1)


    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        return_att: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention values (before normalization) in addition to `Y_logits_pred`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
        """

        X = self.feat_ext(X) # (batch_size, bag_size, feat_dim)

        cls_token = self.cls_token.repeat(X.size(0), 1, 1)
        X = torch.cat([cls_token, X], dim=1) # (batch_size, bag_size+1, feat_dim)

        if mask is not None:
            mask = torch.cat([torch.ones(mask.size(0), 1).to(mask.device), mask], dim=1)

        X = self.transformer_encoder(X, mask) # (batch_size, bag_size, feat_dim)

        z = X[:, 0] # (batch_size, feat_dim)

        Y_pred = self.classifier(z) # (batch_size, 1)

        f = torch.ones(X.size(0), X.size(1)).to(X.device)

        if return_att:
            return Y_pred, f
        else:
            return Y_pred
    
    def compute_loss(
        self,
        Y: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value.
        """

        Y_pred = self.forward(X, mask, return_att=False)

        crit_loss = self.criterion(Y_pred.float(), Y.float())
        crit_name = self.criterion.__class__.__name__

        return Y_pred, {crit_name: crit_loss}

    def predict(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        return_inst_pred: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_inst_pred: If `True`, returns instance labels predictions, in addition to bag label predictions.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            y_inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        return self.forward(X, mask, return_att=return_inst_pred)     