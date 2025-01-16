
import torch
import torch.nn as nn

from .modules import GCNConv


class DeepGraphSurv(torch.nn.Module):
    def __init__(self, 
                 input_dim, 
                 num_layers=4,
                 resample=0.0,
                 hidden_dim=512, 
                 att_dim=128,
                 criterion=torch.nn.BCEWithLogitsLoss()
                ):
        super(DeepGraphSurv, self).__init__()
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

        self.att_fc1 = torch.nn.Linear(hidden_dim, att_dim)
        self.att_gcn = GCNConv(att_dim, att_dim, add_self=True, learn_weight=False)
        self.att_fc2 = torch.nn.Linear(att_dim, 1)

        self.classifier = torch.nn.Linear(hidden_dim, 1)
        self.criterion = criterion

    def forward(self, X, adj_mat, mask, *args, return_att=False, **kwargs):
        
        X = self.fc(X)
        Y = X
        for layer in self.layers:
            Y = layer(Y, adj_mat, mask)
        
        H = self.att_fc1(Y) # (batch_size, bag_size, att_dim)
        H = self.att_gcn(H, adj_mat, mask) # (batch_size, bag_size, att_dim)
        att = self.att_fc2(H) # (batch_size, bag_size, 1)
        att_norm = torch.nn.functional.softmax(att, dim=1) # (batch_size, bag_size, 1)

        z = torch.bmm(Y.transpose(1,2), att_norm).squeeze(dim=2) # (batch_size, hidden_dim)
        
        logits = self.classifier(z).squeeze(1) # (batch_size,)

        if return_att:
            return logits, att.squeeze(2)
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
            loss_dict: dict {'BCEWithLogitsLoss'}
        """
        T_logits_pred = self.forward(X, adj_mat, mask)
        crit_loss = self.criterion(T_logits_pred.float(), T_labels.float())
        crit_name = self.criterion.__class__.__name__
        return T_logits_pred, { crit_name : crit_loss }

    def predict(self, X, adj_mat, mask, *args, return_y_pred=True, **kwargs):
        T_logits_pred, att_val = self.forward(X, adj_mat, mask, return_att=return_y_pred)
        if return_y_pred:
            return T_logits_pred, att_val
        else:
            return T_logits_pred