import torch

from .modules import MILBayesSmoothAttentionPool, MILFeatExt, TransformerEncoder
from .MILModel import MILModel

class TransformerBayesSmoothABMIL(MILModel):
    def __init__(
        self,
        input_shape : tuple,
        feat_ext_name : str = 'none', 
        transformer_encoder_kwargs : dict = {},
        pool_kwargs : dict = {},
        criterion : torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        **kwargs        
        ) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.feat_ext_name = feat_ext_name
        self.pool_kwargs = pool_kwargs
        self.transformer_encoder_kwargs = transformer_encoder_kwargs
        self.kwargs = kwargs
        self.criterion = criterion

        self.feat_ext = MILFeatExt(input_shape=input_shape, feat_ext_name=feat_ext_name)
        self.feat_dim = self.feat_ext.output_size
        self.transformer_encoder = TransformerEncoder(in_dim = self.feat_dim, **transformer_encoder_kwargs)
        self.emb_dim = self.feat_dim
        self.pool = MILBayesSmoothAttentionPool(in_dim = self.emb_dim, **pool_kwargs)
        self.classifier = torch.nn.Linear(self.emb_dim, 1)

    
    def forward(self, X, adj_mat, mask, return_att=False, return_samples=False):
        """
        input:
            X: tensor (batch_size, bag_size, ...)
            adj_mat: sparse coo tensor (batch_size, bag_size, bag_size)
            mask: tensor (batch_size, bag_size)
        output:
            T_logits_pred: tensor (batch_size,)
            att: tensor (batch_size, bag_size) if return_att is True
        """

        X = self.feat_ext(X) # (batch_size, bag_size, feat_dim)
        if self.transformer_encoder is not None:
            Y = self.transformer_encoder(X, adj_mat, mask) # (batch_size, bag_size, feat_dim)
                        
        out_pool = self.pool(Y, adj_mat, mask, return_att=return_att)
        if return_att:
            Z, f = out_pool # Z: (batch_size, D, n_samples), f: (batch_size, bag_size, n_samples)
            if not return_samples:
                f = torch.mean(f, dim=2) # (batch_size, bag_size)
        else:
            Z = out_pool # (batch_size, D, n_samples)
        
        
        Z = Z.transpose(1,2) # (batch_size, n_samples, D)
        T_logits_pred = self.classifier(Z) # (batch_size, n_samples, 1)
        T_logits_pred = T_logits_pred.squeeze(-1) # (batch_size, n_samples)

        if not return_samples:
            T_logits_pred = torch.mean(T_logits_pred, dim=-1) # (batch_size,)

        if return_att:
            return T_logits_pred, f
        else:
            return T_logits_pred
    
    def compute_loss(self, T_labels, X, adj_mat, mask, *args, **kwargs):
        """
        Input:
            T_labels: tensor (batch_size,)
            X: tensor (batch_size, bag_size, ...)
            adj_mat: sparse coo tensor (batch_size, bag_size, bag_size)
            mask: tensor (batch_size, bag_size)
        Output:
            T_logits_pred: tensor (batch_size,)
            loss_dict: dict {'BCEWithLogitsLoss', **pool_loss_dict, ...}
        """

        X = self.feat_ext(X) # (batch_size, bag_size, feat_dim)
        if self.transformer_encoder is not None:
            Y = self.transformer_encoder(X, adj_mat, mask) # (batch_size, bag_size, feat_dim)
        
        Z, pool_loss_dict = self.pool.compute_output_and_loss(T_labels=T_labels, X=Y, adj_mat=adj_mat, mask=mask, *args, **kwargs)
        
        Z = Z.transpose(1,2) # (batch_size, n_samples, emb_dim)
        T_logits_pred = self.classifier(Z) # (batch_size, n_samples, 1)
        T_logits_pred = T_logits_pred.squeeze(-1) # (batch_size, n_samples)
        T_logits_pred_mean = torch.mean(T_logits_pred, dim=-1) # (batch_size,)

        T_labels = T_labels.unsqueeze(-1).expand(-1, T_logits_pred.shape[-1]) # (batch_size, bag_size, n_samples)
        
        crit_loss = self.criterion(T_logits_pred.float(), T_labels.float())
        crit_name = self.criterion.__class__.__name__

        return T_logits_pred_mean, { crit_name: crit_loss, **pool_loss_dict }
    
    @torch.no_grad()
    def predict(self, X, adj_mat, mask, *args, return_y_pred=True, return_samples=False, **kwargs):
        T_logits_pred, att_val = self.forward(X, adj_mat, mask, return_att=return_y_pred, return_samples=return_samples)
        return T_logits_pred, att_val
        


        
        