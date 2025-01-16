import torch
import numpy as np

from .MILModel import MILModel
from .modules import MILFeatExt, MILAttentionPool


def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps

class DFTDMIL(MILModel):
    def __init__(
        self, 
        input_shape,
        feat_ext_name, 
        att_dim : int,
        num_groups : int,
        distill : str,
        criterion : torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ):
        super(DFTDMIL, self).__init__()
        self.input_shape = input_shape
        self.feat_ext_name = feat_ext_name
        self.att_dim = att_dim
        self.num_groups = num_groups
        self.distill = distill
        self.criterion = criterion

        self.feat_ext = MILFeatExt(input_shape=input_shape, feat_ext_name=feat_ext_name)
        self.feat_dim = self.feat_ext.output_size

        self.attention = MILAttentionPool(in_dim = self.feat_dim, att_dim = self.att_dim)
        self.classifier = torch.nn.Linear(self.feat_dim, 1)

        self.u_attention = MILAttentionPool(in_dim = self.feat_dim, att_dim = self.att_dim)
        self.u_classifier = torch.nn.Linear(self.feat_dim, 1)

    def forward(self, X, *args, return_inst_cam = False, **kwargs):
        """
        Input:
            X: tensor (batch_size, bag_size, input_dim)
        Output:
            T_logits: tensor (batch_size, )
            y_logits: tensor (batch_size, bag_size,)
        """

        batch_size = X.size(0)
        # if batch_size > 1:
        #     raise ValueError("[DFTDMIL] Batch size must be 1.")            
        
        num_inst_per_group = X.size(1) // self.num_groups

        while num_inst_per_group < 1.0:
            self.num_groups -= 1
            num_inst_per_group = X.size(1) // self.num_groups

        bag_index = np.arange(0, X.size(1))
        np.random.shuffle(bag_index)
        bag_chunks = np.array_split(bag_index, self.num_groups)

        pseudo_pred_list = []
        pseudo_feat_list = []
        inst_pred_logits_list = []

        for bag_chunk in bag_chunks:
            X_chunk = X[:, bag_chunk, :]
            H = self.feat_ext(X_chunk) # [batch_size, chunk_size, feat_dim]

            z = self.attention(H) # [batch_size, feat_dim], [batch_size, chunk_size]

            # att_val = self.attention(H) # [batch_size, chunk_size, 1]
            # att_norm = F.softmax(att_val, dim=1) # [batch_size, chunk_size, 1]

            # # H_att = torch.einsum('bcm,bcl->bcm', H, att_norm) # [batch_size, chunk_size, feat_dim]
            # # z = torch.sum(H_att, dim=1) # [batch_size, feat_dim]
            # z = torch.bmm(H.permute(0, 2, 1), att_norm) # [batch_size, feat_dim, 1]
            # z = z.squeeze(-1) # [batch_size, feat_dim]

            T_logits_pred = self.classifier(z) # [batch_size, 1]
            pseudo_pred_list.append(T_logits_pred)

            inst_pred_logits = get_cam_1d(self.classifier, H) # [batch_size, 1, chunk_size]
            inst_pred_logits = inst_pred_logits.squeeze(1) # [batch_size, chunk_size]
            inst_pred_logits_list.append(inst_pred_logits)

            _, sort_idx = torch.sort(inst_pred_logits, 1, descending=True) # [batch_size, chunk_size], [batch_size, chunk_size]
            topk_idx_max = sort_idx[:, :num_inst_per_group].long() # [batch_size, num_inst_per_group]
            topk_idx_min = sort_idx[:, -num_inst_per_group:].long() # [batch_size, num_inst_per_group]
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=1) # [batch_size, 2*num_inst_per_group]

            if self.distill == 'maxmin':
                # pseudo_feat = torch.index_select(H, dim=1, index=topk_idx) # [batch_size, 2*num_inst_per_group, feat_dim]
                # print(topk_idx[0,:].max(), topk_idx[0,:].min())
                # print(H.shape)
                # pseudo_feat = H[topk_idx, :]3
                index = topk_idx.unsqueeze(-1).expand(-1, -1, self.feat_dim) # [batch_size, 2*num_inst_per_group, feat_dim]
                pseudo_feat = torch.gather(H, 1, index) # [batch_size, 2*num_inst_per_group, feat_dim]
            elif self.distill == 'max':
                index = topk_idx_max.unsqueeze(-1).expand(-1, -1, self.feat_dim) # [batch_size, num_inst_per_group, feat_dim]
                pseudo_feat = torch.gather(H, 1, index) # [batch_size, num_inst_per_group, feat_dim]
            elif self.distill == 'afs':
                pseudo_feat = z.unsqueeze(1) # [batch_size, 1, feat_dim]

            pseudo_feat_list.append(pseudo_feat)
        
        pseudo_pred = torch.cat(pseudo_pred_list, dim=1) # [batch_size, num_groups]
        pseudo_feat = torch.cat(pseudo_feat_list, dim=1) # [batch_size, num_groups, k, feat_dim]
        pseudo_feat = pseudo_feat.view(batch_size, -1, self.feat_dim) # [batch_size, num_groups*k, feat_dim]
        
        pseudo_z = self.u_attention(pseudo_feat) # [batch_size, feat_dim]
        T_logits_pred = self.u_classifier(pseudo_z) # [batch_size, 1]
        T_logits_pred = T_logits_pred.squeeze(-1) # [batch_size,]

        if return_inst_cam:
            inst_pred_logits = torch.cat(inst_pred_logits_list, dim=1) # [batch_size, bag_size]
            
            # recover the original order
            inst_pred_logits_reorder = torch.zeros_like(inst_pred_logits)
            bag_chunks_idx = np.concatenate(bag_chunks)
            inst_pred_logits_reorder[:, bag_chunks_idx] = inst_pred_logits
            inst_pred_logits = inst_pred_logits_reorder

            return T_logits_pred, pseudo_pred, inst_pred_logits

        return T_logits_pred, pseudo_pred
    
    def compute_loss(self, T_labels, X, *args, **kwargs):
        """
        Input:
            X: tensor (batch_size, bag_size, input_dim)
        Output:
            T_logits: tensor (batch_size, )
            y_logits: tensor (batch_size, bag_size,)
        """
        T_logits_pred, pseudo_pred = self.forward(X, *args, **kwargs) # [batch_size,], [batch_size, num_groups]
        crit_name = self.criterion.__class__.__name__
        crit_loss_t1 = self.criterion(T_logits_pred, T_labels.float())
        T_labels_repeat = T_labels.unsqueeze(1).repeat(1, self.num_groups) # [batch_size, num_groups]
        crit_loss_t2 = self.criterion(pseudo_pred, T_labels_repeat.float()).mean() # [batch_size, num_groups]
        return T_logits_pred, { f'{crit_name}_t1': crit_loss_t1, f'{crit_name}_t2': crit_loss_t2 }
    
    @torch.no_grad()
    def predict(self, X, *args, return_y_pred=True, **kwargs):
        T_logits_pred, pseudo_pred, y_logits_pred = self.forward(X, *args, return_inst_cam=True, **kwargs)
        return T_logits_pred, y_logits_pred



