
# Code is from:
# - CLAM repository: https://github.com/mahmoodlab/CLAM/

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.topk import SmoothTop1SVM
from .modules import MILFeatExt, MILAttentionPool

class CLAM_SB(nn.Module):
    def __init__(
            self, 
            input_shape,
            feat_ext_name, 
            att_dim = 256, 
            dropout = 0.0, 
            k_sample=10, 
            instance_loss_name='svm',
            criterion = torch.nn.BCEWithLogitsLoss()
        ):
        super().__init__()
        self.input_shape = input_shape
        self.feat_ext_name = feat_ext_name
        self.att_dim = att_dim
        self.dropout = dropout
        self.k_sample = k_sample
        self.instance_loss_name = instance_loss_name
        self.criterion = criterion

        self.feat_ext = MILFeatExt(input_shape=input_shape, feat_ext_name=feat_ext_name)
        self.feat_dim = self.feat_ext.output_size

        self.att_pool = MILAttentionPool(in_dim = self.feat_dim, att_dim = self.att_dim)

        self.classifier = nn.Linear(self.feat_dim, 1)
        self.instance_classifiers = nn.ModuleList([nn.Linear(self.feat_dim, 2) for i in range(2)])
        if instance_loss_name == 'svm':
            
            self.instance_loss_fn =  SmoothTop1SVM(n_classes = 2)
        elif instance_loss_name == 'ce':
            self.instance_loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Invalid instance loss function")
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        k_sample = min(self.k_sample, A.shape[0])
        
        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k_sample, device)
        n_targets = self.create_negative_targets(k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        k_sample = min(self.k_sample, A.shape[0])

        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets
    
    def compute_inst_loss(self, A, h, T_label):
        """
        Input:
            A: tensor (batch_size, bag_size,)
            h: tensor (batch_size, bag_size, L)
            T_label: tensor (batch_size,)
        Output:
            inst_loss: float
        """

        sum_inst_loss = 0.0
        batch_size = A.shape[0]
        for i in range(batch_size):
            label = int(T_label[i].item())
            if label == 0:
                in_idx = 0
                out_idx = 1
            else:
                in_idx = 1
                out_idx = 0    
            inst_loss_in, _, _ = self.inst_eval(A[i], h[i], self.instance_classifiers[in_idx])
            inst_loss_out, _, _ = self.inst_eval_out(A[i], h[i], self.instance_classifiers[out_idx])

            sum_inst_loss += inst_loss_in + inst_loss_out
        return sum_inst_loss


    def forward(self, x, *args, return_att=False, return_emb=False, **kwargs):

        x = self.feat_ext(x) # (batch_size, bag_size, D)

        Z, f = self.att_pool(x, *args, return_att=True) # Z: (batch_size, D), f: (batch_size, bag_size)

        T_logits = self.classifier(Z).squeeze(1) # [batch_size,]
        
        if return_emb:
            if return_att:
                return T_logits, f, x
            else:
                return T_logits, x
        elif return_att:
            return T_logits, f
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
        T_logits_pred, att, emb = self.forward(X, *args, return_att = True, return_emb=True, **kwargs)
        crit_loss = self.criterion(T_logits_pred.float(), T_labels.float())
        crit_name = self.criterion.__class__.__name__
        inst_loss = self.compute_inst_loss(att, emb, T_labels)

        return T_logits_pred, { crit_name: crit_loss, 'InstLoss' : inst_loss}

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