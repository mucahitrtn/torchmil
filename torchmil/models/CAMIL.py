import torch
import torch.nn as nn

from .modules.NystromTransformer import NystromTransformerLayer
from .modules import MILFeatExt

from .MILModel import MILModel


class CAMILSelfAttention(nn.Module):
    def __init__(self, in_dim, att_dim=64):
        super(CAMILSelfAttention, self).__init__()
        self.att_dim = att_dim
        self.qk_nn = torch.nn.Linear(in_dim, 2*att_dim, bias = False)
        self.v_nn = torch.nn.Linear(in_dim, in_dim, bias = False)

    def forward(self, x, adj_matrix):
        """
        input:
            x: (batch_size, bag_size, in_dim)
            adj_mat: sparse coo tensor (batch_size, bag_size, bag_size)
        output:
            y: (batch_size, bag_size, in_dim)
        """
        
        q, k = self.qk_nn(x).chunk(2, dim=-1) # (batch_size, bag_size, att_dim), (batch_size, bag_size, att_dim)
        v = self.v_nn(x) # (batch_size, bag_size, att_dim)

        # att_matrix = torch.matmul(q, k.transpose(-2, -1)) / (self.att_dim**0.5) # (batch_size, bag_size, bag_size)
        
        # att_matrix = att_matrix * adj_matrix # (batch_size, bag_size, bag_size)
        
        # w = torch.sum(att_matrix, dim=-1, keepdim=True) # (batch_size, bag_size, 1)
        # w = torch.softmax(w, dim=0) # (batch_size, bag_size, 1)

        inv_scale = 1.0 / (self.att_dim**0.5)

        if adj_matrix.is_sparse:
            adj_matrix = adj_matrix.to_dense()

        w = torch.sum(
            inv_scale * torch.matmul(q, k.transpose(-2, -1)) * adj_matrix,
            dim = -1, keepdim = True
        ) # (batch_size, bag_size, 1)
        
        # q = LazyTensor(q.unsqueeze(-2))  # (batch_size, bag_size, 1, att_dim)
        # k = LazyTensor(k.unsqueeze(-3)) # (batch_size, 1, bag_size, att_dim)
        # adj_matrix = LazyTensor(adj_matrix.unsqueeze(-1)) # (batch_size, bag_size, bag_size, 1)

        # K_mat = (q | k).sum(-1) * inv_scale * adj_matrix # (batch_size, bag_size, bag_size)

        # w = K_mat.sum(-1, keepdim=True) # (batch_size, bag_size, 1)
        
        w = torch.softmax(w, dim=1)  # (batch_size, bag_size, 1)

        # w = torch.sum(att_matrix, dim=-1, keepdim=True) # (batch_size, bag_size, 1)

        l = w*v # (batch_size, bag_size, att_dim)

        return l

class CAMILAttentionPool(nn.Module):
    def __init__(self, in_dim, att_dim=50):
        super(CAMILAttentionPool, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, att_dim)
        self.fc2 = torch.nn.Linear(att_dim, 1, bias=False)

    def forward(self, t, m, mask, return_att=False):
        """
        input:
            t: (batch_size, bag_size, in_dim)
            m: (batch_size, bag_size, d_dim)
            mask: (batch_size, bag_size)
        output:
            z: (batch_size, d_dim)
            att: (batch_size, bag_size) if return_att is True
        """

        mask = mask.unsqueeze(dim=-1) # (batch_size, bag_size, 1)
        
        f = self.fc2(torch.nn.functional.tanh(self.fc1(t))) # (batch_size, bag_size, 1)

        exp_f = torch.exp(f)*mask # (batch_size, bag_size, 1)
        sum_exp_f = torch.sum(exp_f, dim=1, keepdim=True) # (batch_size, 1, 1)
        a = exp_f/sum_exp_f # (batch_size, bag_size, 1)

        z = torch.bmm(m.transpose(1,2), a).squeeze(dim=2) # (batch_size, d_dim)

        if return_att:
            return z, f.squeeze(dim=2)
        else:
            return z

class CAMIL(MILModel):
    def __init__(
            self, 
            input_dim,
            emb_dim = 512,
            att_dim = 128,
            num_heads = 8,
            criterion : torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        ):
        super(CAMIL, self).__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.att_dim = att_dim

        self.feat_ext = nn.Linear(input_dim, emb_dim)

        self.nystrom_transformer_layer = NystromTransformerLayer(dim=self.emb_dim, dim_head=self.att_dim//num_heads, heads=num_heads)
        
        self.camil_self_attention = CAMILSelfAttention(in_dim=self.emb_dim, att_dim=self.att_dim)
        self.camil_att_pool = CAMILAttentionPool(in_dim=self.emb_dim, att_dim=self.att_dim)

        self.class_layer = nn.Linear(self.emb_dim, 1)

        self.criterion = criterion
    
    def forward(self, X, adj_mat, mask, *args, return_att=False, **kwargs):
        """
        Input:
            X: tensor (batch_size, bag_size, input_dim)
            adj_mat: tensor (batch_size, bag_size, bag_size)
            mask: (batch_size, bag_size)
        output:
            T_logits: (batch_size, 1)
            att: (batch_size, bag_size) if return_att is True
        """
        
        # device = X.device
        # batch_size, bag_size = X.shape[0], X.shape[1]

        h = self.feat_ext(X) # (batch_size, bag_size, 512)
        t = self.nystrom_transformer_layer(h) # [batch_size, bag_size, 512]

        l = self.camil_self_attention(t, adj_mat) # [batch_size, bag_size, 512]

        m = torch.sigmoid(l)*l + (1 - torch.sigmoid(l))*t # [batch_size, bag_size, 512]
        # m = self._sigmoid(l)*l + (1.0 - self._sigmoid(l))*t # [batch_size, bag_size, 512]

        if return_att:
            z, attn_val = self.camil_att_pool(t, m, mask, return_att=True) # [batch_size, 512], [batch_size, bag_size]
        else:
            z = self.camil_att_pool(t, m, mask)

        T_logits = self.class_layer(z).squeeze(dim=1)  # [batch_size,]
        
        if return_att:
            return T_logits, attn_val
        else:
            return T_logits
    
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
    
    @torch.no_grad()
    def predict(self, X, adj_mat, mask, *args, return_y_pred=True, **kwargs):
        T_logits_pred, att_val = self.forward(X, adj_mat, mask, return_att=return_y_pred)
        if return_y_pred:
            return T_logits_pred, att_val
        else:
            return T_logits_pred