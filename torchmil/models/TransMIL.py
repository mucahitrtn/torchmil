import torch
import torch.nn as nn
import numpy as np

from .modules.NystromTransformer import NystromTransformerLayer

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:] # cls_token: [B, 512], feat_token: [B, H*W, 512]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W) # [B, 512, H, W]
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat) # [B, 512, H, W]
        x = x.flatten(2).transpose(1, 2) # [B, H*W, 512]
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1) # [B, H*W+1, 512]
        return x


class TransMIL(torch.nn.Module):
    def __init__(
            self, 
            in_dim : int,
            emb_dim : int,
            n_heads = 4,
            n_landmarks = None,
            pinv_iterations = 6,
            residual = True,
            dropout = 0.0,
            use_mlp = False,
            criterion : torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        ):
        """
        Arguments:
            in_dim: Input dimension.
            emb_dim: Embedding dimension.
            n_heads: Number of heads.
            n_landmarks: Number of landmarks.
            pinv_iterations: Number of iterations for the pseudo-inverse.
            residual: Whether to use residual in the transformer attention layer.
            dropout: Dropout rate.
            use_mlp: Whether to use a MLP after the transformer attention layer.
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits.        
        """

        super(TransMIL, self).__init__()

        if n_landmarks is None:
            n_landmarks = emb_dim//2
        self.n_landmarks = n_landmarks

        self.feat_ext = nn.Linear(in_dim, emb_dim)

        self.pos_layer = PPEG(dim=emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        self.transf_layer_1 = NystromTransformerLayer(att_dim=emb_dim, n_heads=n_heads, n_landmarks=n_landmarks, pinv_iterations=pinv_iterations, residual=residual, dropout=dropout, use_mlp=use_mlp)
        self.transf_layer_2 = NystromTransformerLayer(att_dim=emb_dim, n_heads=n_heads, n_landmarks=n_landmarks, pinv_iterations=pinv_iterations, residual=residual, dropout=dropout, use_mlp=use_mlp)

        self.norm = nn.LayerNorm(emb_dim)
        self.classifier = nn.Linear(emb_dim, 1)

        self.criterion = criterion

    def forward(
            self, 
            X : torch.Tensor,
            return_att : bool = False
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.
            return_att: Whether to return the attention matrix.
        
        Returns:
            bag_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
        """
        
        device = X.device
        batch_size, bag_size = X.shape[0], X.shape[1]

        X = self.feat_ext(X) # (batch_size, bag_size, emb_dim)

        # pad
        bag_size = X.shape[1]
        padded_size = int(np.ceil(np.sqrt(bag_size)))
        add_length = padded_size*padded_size - bag_size
        X = torch.cat([X, X[:,:add_length,:]], dim = 1) # (batch_size, padded_size*padded_size, emb_dim)

        # add cls_token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(device) # (batch_size, 1, emb_dim)
        X = torch.cat((cls_tokens, X), dim=1) # (batch_size, padded_size*padded_size+1, emb_dim)

        # transformer layer

        X = self.transf_layer_1(X) # (batch_size, padded_size*padded_size+1, emb_dim)

        # pos layer
        X = self.pos_layer(X, padded_size, padded_size) # (batch_size, padded_size*padded_size+1, emb_dim)
        
        # transformer layer
        if return_att:
            current_len = padded_size*padded_size+1
            pad_len = self.n_landmarks - (current_len % self.n_landmarks) if current_len % self.n_landmarks > 0 else 0
            X, attn = self.transf_layer_2(X, return_attn=True) # (batch_size, padded_size*padded_size+1, emb_dim), (batch_size, n_heads, padded_size*padded_size+1, padded_size*padded_size+1)
            attn_mat = attn[:, :, pad_len:pad_len+bag_size+1, pad_len:pad_len+bag_size+1] # (batch_size, n_heads, bag_size+1, bag_size+1)
            attn_mat = attn_mat.mean(dim=1) # (batch_size, bag_size+1, bag_size+1)
            att = attn_mat[:, 0, 1:] # (batch_size, bag_size)
        else:
            X = self.transf_layer_2(X) # (batch_size, padded_size*padded_size+1, emb_dim)

        # norm layer
        X = self.norm(X) # (batch_size, padded_size*padded_size+1, emb_dim)

        # cls_token
        cls_token = X[:,0] # (batch_size, emb_dim)

        # predict
        bag_pred = self.classifier(cls_token).squeeze(-1) # (batch_size,)

        if return_att:
            return bag_pred, att
        else:
            return bag_pred
    
    def compute_loss(
        self,
        labels: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
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