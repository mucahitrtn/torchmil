import torch
import torch.nn as nn
import numpy as np

from .MILModel import MILModel
from .modules import MILFeatExt
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


class TransMIL(MILModel):
    def __init__(
            self, 
            input_dim,
            emb_dim = 512,
            att_dim = 128,
            num_heads = 8,
            criterion : torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
            **kwargs
        ):
        super(TransMIL, self).__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.att_dim = att_dim

        self.feat_ext = nn.Linear(input_dim, emb_dim)

        self.pos_layer = PPEG(dim=self.emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_dim))

        self.transf_layer_1 = NystromTransformerLayer(dim=self.emb_dim, dim_head=self.att_dim//num_heads, heads=num_heads)
        self.transf_layer_2 = NystromTransformerLayer(dim=self.emb_dim, dim_head=self.att_dim//num_heads, heads=num_heads)

        self.norm = nn.LayerNorm(self.emb_dim)
        self.classifier = nn.Linear(self.emb_dim, 1)

        self.criterion = criterion

    def forward(self, X, *args, return_att=False, **kwargs):
        """
        Input:
            X : tensor (batch_size, bag_size, input_dim)
        Output:
            T_logits : tensor (batch_size,)
            att : tensor (batch_size, bag_size) if return_att is True
        """
        
        device = X.device
        batch_size, bag_size = X.shape[0], X.shape[1]

        h = self.feat_ext(X) # (batch_size, bag_size, 512)

        # pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]], dim = 1) # (batch_size, _H*_W, 512)

        # add cls_token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(device) # (batch_size, 1, 512)
        h = torch.cat((cls_tokens, h), dim=1) # (batch_size, _H*_W+1, 512)

        # transformer layer

        h = self.transf_layer_1(h) # (batch_size, _H*_W+1, 512)

        # pos layer
        h = self.pos_layer(h, _H, _W) # (batch_size, _H*_W+1, 512)
        
        # transformer layer
        if return_att:
            num_landmarks = 512//2
            current_len = _H*_W+1
            pad_len = num_landmarks - (current_len % num_landmarks) if current_len % num_landmarks > 0 else 0
            h, attn = self.transf_layer_2(h, return_attn=True) # [B, _H*_W+1, 512], [B, 8, pad, pad]
            attn_mat = attn[:, :, pad_len:pad_len+bag_size+1, pad_len:pad_len+bag_size+1] # [B, 8, bag_size+1, bag_size+1]
            attn_mat = attn_mat.mean(dim=1) # [B, bag_size+1, bag_size+1]
            att = attn_mat[:, 0, 1:] # [B, bag_size]
        else:
            h = self.transf_layer_2(h)

        # norm layer
        h = self.norm(h) # (batch_size, _H*_W+1, 512)

        # cls_token
        h = h[:,0] # (batch_size, 512)

        # predict
        T_logits = self.classifier(h).squeeze(dim=1) # (batch_size,)

        if return_att:
            return T_logits, att
        else:
            return T_logits
    
    def compute_loss(self, T_labels, X, *args, **kwargs):
        """
        Input:
            T_labels: tensor (batch_size,)
            X: tensor (batch_size, bag_size, ...)
        Output:
            T_logits_pred: tensor (batch_size,)
            loss_dict: dict {'BCEWithLogitsLoss'}
        """
        T_logits_pred = self.forward(X, *args, **kwargs)
        crit_loss = self.criterion(T_logits_pred.float(), T_labels.float())
        crit_name = self.criterion.__class__.__name__
        return T_logits_pred, { crit_name : crit_loss }

    @torch.no_grad()
    def predict(self, X, *args, return_y_pred=True, **kwargs):
        """
        Input:
            X: tensor (batch_size, bag_size, ...)
        Output:
            T_logits_pred: tensor (batch_size,)
            y_pred: tensor (batch_size, bag_size) if return_y_pred is True
        """
        return self.forward(X, *args, return_att=return_y_pred, **kwargs)