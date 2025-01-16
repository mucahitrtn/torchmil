
import torch
import torch.nn.functional as F
import torch.nn as nn

from .modules.GCNConv import GCNConv
from torchmil.models.modules.attention_pool import MILAttentionPool  

class PathGCN(torch.nn.Module):
    def __init__(self, 
                 input_dim, 
                 num_layers=4,
                 resample=0,
                 hidden_dim=512, 
                 att_dim=128,
                 criterion=torch.nn.BCEWithLogitsLoss()
                ):
        super(PathGCN, self).__init__()
        self.num_layers = num_layers
        self.resample = resample

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample), nn.Linear(input_dim, hidden_dim), nn.ReLU()])
        else:
            self.fc = nn.Sequential(*[nn.Linear(input_dim, hidden_dim), nn.ReLU()])

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            layer = GCNConv(hidden_dim, hidden_dim, add_self=True, learn_weight=False)
            self.layers.append(layer)

        self.attention = MILAttentionPool(in_dim = hidden_dim, att_dim = att_dim)
        self.classifier = torch.nn.Linear(hidden_dim, 1)    
        self.criterion = criterion

    def forward(self, X, adj_mat, mask, *args, return_att=False, **kwargs):
        
        x = self.fc(X) # (batch_size, bag_size, hidden_dim)
        x_ = x 
        for layer in self.layers:
            x = layer(x, adj_mat, mask) # (batch_size, bag_size, hidden_dim)
            x_ = torch.cat([x_, x], axis=1) # (batch_size, bag_size*(i+2), hidden_dim)
        
        if return_att:
            z, att = self.attention(x_, return_att=True) # (batch_size, hidden_dim)
        else:
            z = self.attention(x_) # (batch_size, hidden_dim)
        
        logits = self.classifier(z).squeeze(1) # (batch_size,)

        if return_att:
            bag_size = X.shape[1]
            att = att[:, :bag_size]
            return logits, att
        else:
            return logits        

    def compute_loss(self, T_labels, X, adj_mat, mask, *args, **kwargs):
        """
        Input:
            T_labels: tensor (batch_size,)
            X: tensor (batch_size, bag_size, ...)
            adj_mat: tensor (batch_size, bag_size, bag_size)
            mask: tensor (batch_size, bag_size)
        Output:
            T_logits_pred: tensor (batch_size,)
            loss_dict: dict {crit_name: crit_loss}
        """
        T_logits_pred = self.forward(X, adj_mat, mask)
        crit_loss = self.criterion(T_logits_pred.float(), T_labels.float())
        crit_name = self.criterion.__class__.__name__
        return T_logits_pred, { crit_name: crit_loss }

    def predict(self, X, adj_mat, mask, *args, return_y_pred=True, **kwargs):
        T_logits_pred, att_val = self.forward(X, adj_mat, mask, return_att=return_y_pred)
        if return_y_pred:
            return T_logits_pred, att_val
        else:
            return T_logits_pred