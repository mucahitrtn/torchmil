from typing import Union

import torch

from .mil_model import MILModel

from torchmil.nn import SmAttentionPool, SmTransformerEncoder

from torchmil.nn.utils import get_feat_dim

class SmTransformerABMIL(MILModel):
    """
    Transformer Attention-based Multiple Instance Learning (ABMIL) model with the Sm operator. 
    """
    def __init__(
        self,
        in_shape : tuple,
        pool_att_dim : int = 128,
        pool_act : str = 'tanh',
        pool_sm_mode : str = 'approx',
        pool_sm_alpha : Union[float, str] = 'trainable',
        pool_sm_layers : int = 1,
        pool_sm_steps : int = 10,
        pool_sm_pre : bool = False,
        pool_sm_post : bool = False,
        pool_sm_spectral_norm : bool = False,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        transf_att_dim : int = 512, 
        transf_n_layers : int = 1,
        transf_n_heads : int = 4,
        transf_use_mlp : bool = True,
        transf_add_self : bool = True,
        transf_dropout : float = 0.0,
        transf_sm_alpha: float = None,
        transf_sm_mode: str = None,
        transf_sm_steps: int = 10,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        ) -> None:
        """
        Class constructor.

        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension).
            pool_att_dim: Attention dimension for pooling.
            pool_act: Activation function for pooling. Possible values: 'tanh', 'relu', 'gelu'.
            pool_sm_mode: Mode for the Sm operator in pooling. Possible values: 'approx', 'exact'.
            pool_sm_alpha: Alpha value for the Sm operator in pooling. If 'trainable', alpha is trainable.
            pool_sm_layers: Number of layers that use the Sm operator in pooling.
            pool_sm_steps: Number of steps for the Sm operator in pooling.
            pool_sm_pre: If True, apply Sm operator before the attention pooling.
            pool_sm_post: If True, apply Sm operator after the attention pooling.
            pool_sm_spectral_norm: If True, apply spectral normalization to all linear layers in pooling.
            feat_ext: Feature extractor.
            transf_att_dim: Attention dimension for transformer encoder.
            transf_n_layers: Number of layers in transformer encoder.
            transf_n_heads: Number of heads in transformer encoder.
            transf_use_mlp: Whether to use MLP in transformer encoder.
            transf_add_self: Whether to add input to output in transformer encoder.
            transf_dropout: Dropout rate in transformer encoder.
            transf_sm_alpha: Alpha value for the Sm operator in transformer encoder.
            transf_sm_mode: Mode for the Sm operator in transformer encoder.
            transf_sm_steps: Number of steps for the Sm operator in transformer encoder.
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits for binary classification.
        """
        super().__init__()
        self.criterion = criterion

        self.feat_ext = feat_ext
        feat_dim = get_feat_dim(feat_ext, in_shape)
        self.transformer_encoder = SmTransformerEncoder(
            in_dim=feat_dim, 
            att_dim=transf_att_dim, 
            n_layers=transf_n_layers, 
            n_heads=transf_n_heads, 
            use_mlp=transf_use_mlp,
            add_self=transf_add_self, 
            dropout=transf_dropout,
            sm_alpha=transf_sm_alpha,
            sm_mode=transf_sm_mode,
            sm_steps=transf_sm_steps
        )
        self.pool = SmAttentionPool(
            in_dim=feat_dim, 
            att_dim=pool_att_dim, 
            act=pool_act, 
            sm_mode=pool_sm_mode, 
            sm_alpha=pool_sm_alpha, 
            sm_layers=pool_sm_layers, 
            sm_steps=pool_sm_steps, 
            sm_pre=pool_sm_pre, 
            sm_post=pool_sm_post, 
            sm_spectral_norm=pool_sm_spectral_norm
        )
        self.last_layer = torch.nn.Linear(feat_dim, 1)

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

        X = self.feat_ext(X) # (batch_size, bag_size, feat_dim)

        Y = self.transformer_encoder(X, mask=mask, adj=adj) # (batch_size, bag_size, feat_dim)

        out_pool = self.pool(Y, adj=adj, mask=mask, return_att=return_att)
        if return_att:
            Z, f = out_pool # Z: (batch_size, emb_dim), f: (batch_size, bag_size)
        else:
            Z = out_pool # (batch_size, emb_dim)
        
        Y_pred = self.last_layer(Z) # (batch_size, n_samples, 1)
        Y_pred = Y_pred.squeeze(-1) # (batch_size,)

        if return_att:
            return Y_pred, f
        else:
            return Y_pred
    
    def compute_loss(
        self,
        Y: torch.Tensor,
        X: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value.
        """

        Y_pred = self.forward(X, mask=mask, adj=adj, return_att=False)

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
            mask: Mask of shape `(batch_size, bag_size)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            return_inst_pred: If `True`, returns instance labels predictions, in addition to bag label predictions.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            y_inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        return self.forward(X, adj, mask, return_att=return_inst_pred)
        


        
        