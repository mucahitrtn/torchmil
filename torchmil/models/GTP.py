import torch
import numpy as np

from .modules.GTPTransformer import VisionTransformer, Linear

from .modules.GCNConv import GCNConv
from .modules.dense_mincut_pool import dense_mincut_pool


class GTP(torch.nn.Module):
    def __init__(self, 
                 input_dim, 
                 num_layers = 2,
                 embed_dim = 64,
                 num_heads = 4,
                 node_cluster_num = 100,
                 criterion : torch.nn.Module = torch.nn.BCEWithLogitsLoss()
        ):

        super(GTP, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.node_cluster_num = node_cluster_num
        self.num_heads = num_heads
        self.n_class = 1

        self.transformer = VisionTransformer(num_classes=self.n_class, embed_dim=self.embed_dim, depth=self.num_layers, num_heads=self.num_heads)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.embed_dim)) # (1, 1, embed_dim)
        self.criterion = criterion

        self.dense_mincut_pool = dense_mincut_pool()

        self.fc1 = Linear(self.input_dim, self.embed_dim)

        self.conv1 = GCNConv(self.embed_dim, self.embed_dim, add_self=True, learn_weight=True)
        self.pool1 = Linear(self.embed_dim, self.node_cluster_num)

    def forward(self, X, adj_mat, mask, *args, return_cam=False, return_loss = False, **kwargs):
        """
        input:
            X: (batch_size, bag_size, input_dim)
            adj_mat: (batch_size, bag_size, bag_size)
            mask: (batch_size, bag_size)
        output:
            T_logits: (batch_size, 1)
        """
        
        batch_size, _, _ = X.shape
                
        X = self.fc1(X) # (batch_size, bag_size, embed_dim)
        X = self.conv1(X, adj_mat, mask) # (batch_size, bag_size, embed_dim)
        s = self.pool1(X) # (batch_size, bag_size, node_cluster_num)
                
        X, adj_mat, mc_loss, o_loss = self.dense_mincut_pool(X, adj_mat, s, mask)   # X: (batch_size, node_cluster_num, input_dim), 
                                                                                    # adj_mat: (batch_size, node_cluster_num, node_cluster_num), 
                                                                                    # mc_loss: (batch_size, 1), 
                                                                                    # o_loss: (batch_size, 1)
        
        cls_token = self.cls_token.repeat(batch_size, 1, 1) # (batch_size, 1, embed_dim)
        X = torch.cat([cls_token, X], dim=1) # (batch_size, node_cluster_num+1, embed_dim)

        out = self.transformer(X)     # (batch_size, n_class)
        T_logits = out.squeeze(-1)  # (batch_size, n_class) -> (batch_size,) if n_class=1

        # GradCam Attributions, only for batch_size=1
        if return_cam:
            if batch_size != 1:
                raise ValueError("[GTP] Batch size should be 1 for GradCam Attributions")

            s_matrix_ori = s[0] # (bag_size, node_cluster_num)
            assign_matrix = torch.nn.functional.softmax(s_matrix_ori, dim=1) # (bag_size, node_cluster_num)
            cam_matrix_list = []
            for index_ in range(self.n_class):
                one_hot = np.zeros((1, out.size()[-1]), dtype=np.float32) # (1, n_class)
                one_hot[0, index_] = out[0][index_] # (1, n_class)
                one_hot_vector = one_hot # (1, n_class)
                one_hot = torch.from_numpy(one_hot).requires_grad_(True) # (1, n_class)
                one_hot = torch.sum(one_hot.to(out.device) * out) # (1,)
                self.transformer.zero_grad()
                one_hot.backward(retain_graph=True)
                cam = self.transformer.relprop(
                                            torch.tensor(one_hot_vector).to(X.device), 
                                            method="transformer_attribution", 
                                            is_ablation=False, 
                                            start_layer=0, 
                                            alpha = 1
                                        ) # (1, node_cluster_num)
                cam_matrix = torch.mm(assign_matrix, cam.transpose(1,0)).squeeze(-1) # (bag_size,)
                cam_matrix_list.append(cam_matrix) 
            
            cam_matrix = torch.stack(cam_matrix_list, dim=1) # (bag_size, n_class)
            cam_matrix = cam_matrix.unsqueeze(0) # (batch_size = 1, bag_size, n_class)
            cam_matrix = cam_matrix.squeeze(-1) # (batch_size = 1, bag_size, ) if n_class=1
            ### NOTICE: the original implementation only allows batch_size=1 to obtain GraphCAM. We have batch_size=1 in test (only moment when we need GraphCAM), so no problem with that.

        if return_loss:
            loss_dict = {'MinCutLoss': mc_loss, 'OrthoLoss': o_loss}
            if return_cam:
                return T_logits, cam_matrix, loss_dict
            else:
                return T_logits, loss_dict
        elif return_cam:
            return T_logits, cam_matrix
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
        T_logits_pred, loss_dict = self.forward(X, adj_mat, mask, return_loss = True)
        crit_loss = self.criterion(T_logits_pred.float(), T_labels.float())
        crit_name = self.criterion.__class__.__name__
        return T_logits_pred, { crit_name : crit_loss, **loss_dict }

    def predict(self, X, adj_mat, mask, *args, return_y_pred=True, **kwargs):
        T_logits_pred, att_val = self.forward(X, adj_mat, mask, return_cam=return_y_pred)
        if return_y_pred:
            return T_logits_pred, att_val
        else:
            return T_logits_pred