import torch

from torchmil.models.modules import AttentionPool, TransformerEncoder

class TransformerABMIL(torch.nn.Module):
    """
    Transformer Attention-based Multiple Instance Learning (ABMIL) model.    
    """
    def __init__(
        self,
        input_shape : tuple,
        pool_att_dim : int = 128,
        pool_act : str = 'tanh',
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        transf_att_dim : int = None, 
        transf_n_layers : int = 1,
        transf_n_heads : int = 8,
        transf_use_mlp : bool = True,
        transf_add_self : bool = True,
        transf_dropout : float = 0.0,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        ) -> None:
        """
        Class constructor.

        Arguments:
            input_shape: Shape of input data expected by the feature extractor (excluding batch dimension).
            pool_att_dim: Attention dimension for pooling.
            pool_act: Activation function for pooling. Possible values: 'tanh', 'relu', 'gelu'.
            feat_ext: Feature extractor.
            transf_att_dim: Attention dimension for transformer encoder.
            transf_n_layers: Number of layers in transformer encoder.
            transf_n_heads: Number of heads in transformer encoder.
            transf_use_mlp: Whether to use MLP in transformer encoder.
            transf_add_self: Whether to add input to output in transformer encoder.
            transf_dropout: Dropout rate in transformer encoder.
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits for binary classification.
        """
        super().__init__()
        self.input_shape = input_shape
        self.criterion = criterion

        self.feat_ext = feat_ext
        feat_dim = self._get_feat_dim()
        self.transformer_encoder = TransformerEncoder(
            in_dim=feat_dim, 
            att_dim=transf_att_dim, 
            n_layers=transf_n_layers, 
            n_heads=transf_n_heads, 
            use_mlp=transf_use_mlp,
            add_self=transf_add_self, 
            dropout=transf_dropout
        )
        self.pool = AttentionPool(in_dim=feat_dim, att_dim=pool_att_dim, act=pool_act)
        self.last_layer = torch.nn.Linear(feat_dim, 1)

    def _get_feat_dim(self) -> int:
        """
        Get feature dimension of the feature extractor.
        """
        with torch.no_grad():
            return self.feat_ext(torch.zeros((1, *self.input_shape))).shape[-1]

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
            Y_logits_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
        """

        X = self.feat_ext(X) # (batch_size, bag_size, feat_dim)

        Y = self.transformer_encoder(X, mask) # (batch_size, bag_size, feat_dim)

        out_pool = self.pool(Y, mask, return_att=return_att)
        if return_att:
            Z, f = out_pool # Z: (batch_size, emb_dim), f: (batch_size, bag_size)
        else:
            Z = out_pool # (batch_size, emb_dim)
        
        bag_pred = self.last_layer(Z) # (batch_size, n_samples, 1)
        bag_pred = bag_pred.squeeze(-1) # (batch_size,)

        if return_att:
            return bag_pred, f
        else:
            return bag_pred
    
    def compute_loss(
        self,
        Y_true: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
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
            bag_pred: Bag label logits of shape `(batch_size,)`.
            inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        return self.forward(X, mask, return_att=return_inst_pred)
        


        
        