import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .MILModel import MILModel
from .modules import MILFeatExt

def batched_index_select(input, dim, index):
    views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)
    
class DSMIL(MILModel):
    def __init__(
            self, 
            input_shape,
            feat_ext_name,
            att_dim = 128,
            num_classes = 1, 
            dropout_v=0.0, 
            nonlinear=True, 
            passing_v=False,
            criterion : torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
            **kwargs
        ):
        super(DSMIL, self).__init__()
        self.input_shape = input_shape
        self.feat_ext_name = feat_ext_name
        self.att_dim = att_dim

        self.num_classes = num_classes
        self.dropout_v = dropout_v
        self.nonlinear = nonlinear
        self.passing_v = passing_v
        self.criterion = criterion

        self.feat_ext = MILFeatExt(input_shape=input_shape, feat_ext_name=feat_ext_name)
        self.feat_dim = self.feat_ext.output_size

        self.inst_classifier = nn.Linear(self.feat_dim, num_classes)

        if self.nonlinear:
            self.q_nn = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim), nn.ReLU(), nn.Linear(self.feat_dim, self.feat_dim), nn.Tanh())
        else:
            self.q_nn = nn.Linear(self.feat_dim, self.feat_dim)
        if self.passing_v:
            self.v_nn = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(self.feat_dim, self.feat_dim),
                nn.ReLU()
            )
        else:
            self.v_nn = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.bag_classifier = nn.Conv1d(self.num_classes, self.num_classes, kernel_size=self.feat_dim)

    def _compute_feat_dim(self, input_shape):
        input = torch.rand(1, *input_shape) # [1, *input_shape]
        output_feat = self.feature_extractor(input) # [1, feat_dim]
        return output_feat.size(-1)

    def forward(self, x, *args, **kwargs):
        """
        Input:
            x: tensor (batch_size, *input_shape)
        Output:
            T_logits: [batch_size, num_classes]
            y_logits: [batch_size, bag_size, num_classes]
        """
        feats = self.feat_ext(x) # [batch_size, bag_size, feat_dim]

        y_logits = self.inst_classifier(feats) # [batch_size, bag_size, num_classes]

        V = self.v_nn(feats) # [batch_size, bag_size, feat_dim]
        Q = self.q_nn(feats) # [batch_size, bag_size, 128]
        
        # handle multiple classes without for loop
        # sort class scores along the instance dimension
        _, m_indices = torch.sort(y_logits, 1, descending=True) # [batch_size, bag_size, num_classes], [batch_size, bag_size, num_classes]
        
        # select critical instances
        m_feats = batched_index_select(feats, 1, m_indices[:, 0, :]) # [batch_size, num_classes, feat_dim]

        # compute queries of critical instances
        q_max = self.q_nn(m_feats) # [batch_size, num_classes, feat_dim]

        # compute inner product of Q to each entry of q_max
        A = torch.bmm(Q, q_max.transpose(1, 2)) # [batch_size, bag_size, num_classes]
        
        # scale and normalize the attention scores
        scale = np.sqrt(self.feat_dim)
        A = F.softmax( A / scale, 1) # [batch_size, bag_size, num_classes]

        # compute bag representation
        B = torch.bmm(A.transpose(1, 2), V) # [batch_size, num_classes, feat_dim]
                
        T_logits = self.bag_classifier(B).squeeze(-1) # [batch_size, num_classes]

        # squeeze for the case num_classes = 1
        T_logits = T_logits.squeeze(-1)
        y_logits = y_logits.squeeze(-1)
        
        return T_logits, y_logits

    def compute_loss(self, T_labels, X, *args, **kwargs):
        T_logits_pred, y_logits_pred = self.forward(X, *args, **kwargs)
        max_pred, _ = torch.max(y_logits_pred, 1) # [batch_size, num_classes]
        crit_loss = self.criterion(T_logits_pred.float(), T_labels.float())
        crit_name = self.criterion.__class__.__name__
        max_loss = self.criterion(max_pred.float(), T_labels.float())
        return T_logits_pred, { crit_name : crit_loss, f'{crit_name}_max': max_loss }
    
    @torch.no_grad()
    def predict(self, X, *args, **kwargs):
        T_logits_pred, y_logits_pred = self.forward(X, *args, **kwargs)
        max_pred, _ = torch.max(y_logits_pred, 1) # [batch_size, num_classes]
        bag_pred = 0.5*(T_logits_pred + max_pred)
        return bag_pred, y_logits_pred