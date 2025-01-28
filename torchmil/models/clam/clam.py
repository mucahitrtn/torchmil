
# Code is from:
# - CLAM repository: https://github.com/mahmoodlab/CLAM/

import torch

import numpy as np

from .utils import SmoothTop1SVM
from torchmil.models.modules import AttentionPool

from torchmil.models.modules.utils import get_feat_dim, LazyLinear

class CLAM_SB(torch.nn.Module):
    r"""
    CLAM

    Code adapted from the [official repository](https://github.com/mahmoodlab/CLAM/). 
    """
    def __init__(
        self, 
        in_shape : tuple = None,
        att_dim : int = 128,
        att_act : str = 'tanh',
        k_sample : int = 10,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss()
    ) -> None:
        """

        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension).
            att_dim: Attention dimension.
            att_act: Activation function for attention. Possible values: 'tanh', 'relu', 'gelu'.
            k_sample: Number of instances to sample.
            feat_ext: Feature extractor.
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits.        
        """
        super().__init__()
        self.criterion = criterion
        self.feat_ext = feat_ext
        self.k_sample = k_sample

        if in_shape is not None:
            feat_dim = get_feat_dim(feat_ext, in_shape)
        else:
            feat_dim = None

        self.pool = AttentionPool(in_dim = feat_dim, att_dim = att_dim, act = att_act)
        self.classifier = LazyLinear(feat_dim, 1)
        self.inst_classifiers = torch.nn.ModuleList([LazyLinear(feat_dim, 2) for i in range(2)])
        self.inst_loss_fn = SmoothTop1SVM(n_classes = 2)
    
    @staticmethod
    def create_positive_targets(length : int, device : torch.device) -> torch.Tensor:
        """
        Create positive targets.

        Arguments:
            length: Length of the target tensor.
            device: Device to create the tensor.
        
        Returns:
            pos_targets: Tensor of shape `(length,)` with all elements set to 1.        
        """
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length : int, device : torch.device) -> torch.Tensor:
        """
        Create negative targets.

        Arguments:
            length: Length of the target tensor.
            device: Device to create the tensor.
        
        Returns:
            neg_targets: Tensor of shape `(length,)` with all elements set to 0.        
        """
        return torch.full((length, ), 0, device=device).long()
    
    def inst_eval(
            self, 
            att : torch.Tensor,
            emb : torch.Tensor,
            classifier : torch.nn.Module
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate instance-level loss for in-the-class attention branch.

        Arguments:
            att: Attention values of shape `(bag_size,)`.
            emb: Embeddings of shape `(bag_size, feat_dim)`.
            classifier: Instance classifier.
        
        Returns:
            instance_loss: Instance loss.
            all_preds: Predicted instance labels of shape `(2 * k_sample,)`.
            all_targets: Generated instance labels of shape `(2 * k_sample,)`.
        """ 
        device = att.device
        bag_size = emb.shape[0]

        k_sample = min(self.k_sample, bag_size)
        
        top_p_ids = torch.topk(att, k_sample)[1] # (k_sample,)
        top_p = torch.index_select(emb, dim=0, index=top_p_ids) # (k_sample, feat_dim)
        top_n_ids = torch.topk(-att, k_sample)[1] # (k_sample,)
        top_n = torch.index_select(emb, dim=0, index=top_n_ids) # (k_sample, feat_dim)
        p_targets = self.create_positive_targets(k_sample, device) # (k_sample,)
        n_targets = self.create_negative_targets(k_sample, device) # (k_sample,)

        all_targets = torch.cat([p_targets, n_targets], dim=0) # (2 * k_sample,)
        all_instances = torch.cat([top_p, top_n], dim=0) # (2 * k_sample, feat_dim)
        logits = classifier(all_instances) # (2 * k_sample, 2)
        all_preds = torch.topk(logits, 1, dim = 1)[1] # (2 * k_sample,)
        instance_loss = self.inst_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # def inst_eval(
    #     self, 
    #     att : torch.Tensor,
    #     emb : torch.Tensor,
    #     classifier : torch.nn.Module
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """

    #     Evaluate instance-level loss for in-the-class attention branch.

    #     Arguments:
    #         att: Attention values of shape `(batch_size, bag_size,)`.
    #         emb: Embeddings of shape `(batch_size, bag_size, feat_dim)`.
    #         classifier: Instance classifier.
        
    #     Returns:
    #         instance_loss: Instance loss.
    #         all_preds: Predicted instance labels of shape `(batch_size, 2 * k_sample,)`.
    #         all_targets: Generated instance labels of shape `(batch_size, 2 * k_sample,)`.
    #     """ 
    #     device = att.device
    #     batch_size, bag_size, feat_dim = emb.shape
    #     k_sample = min(self.k_sample, bag_size)
        
    #     # Get top-k positive and negative indices for the entire batch
    #     top_p_ids = torch.topk(att, k_sample, dim=1)[1] # (batch_size, k_sample)
    #     top_n_ids = torch.topk(-att, k_sample, dim=1)[1] # (batch_size, k_sample)

    #     # Get top-k positive and negative embeddings for the entire batch
    #     top_p = torch.gather(emb, 1, top_p_ids.unsqueeze(-1).expand(-1, -1, feat_dim)) # (batch_size, k_sample, feat_dim)
    #     top_n = torch.gather(emb, 1, top_n_ids.unsqueeze(-1).expand(-1, -1, feat_dim)) # (batch_size, k_sample, feat_dim)

    #     # Create positive and negative targets for the entire batch
    #     p_targets = self.create_positive_targets(k_sample, device) # (k_sample,)
    #     n_targets = self.create_negative_targets(k_sample, device) # (k_sample,)
    #     all_targets = torch.cat([p_targets, n_targets], dim=0).unsqueeze(0).expand(batch_size, -1)  # (batch_size, 2 * k_sample)


    #     # Concatenate positive and negative embeddings for the entire batch
    #     all_instances = torch.cat([top_p, top_n], dim=1) # (batch_size, 2 * k_sample, feat_dim)

    #     # Get logits for the entire batch
    #     logits = classifier(all_instances) # (batch_size, 2 * k_sample, 2)
    #     all_preds = torch.topk(logits, 1, dim = 2)[1].squeeze(-1) # (batch_size, 2 * k_sample)

    #     # Compute instance loss for the entire batch
    #     instance_loss = self.inst_loss_fn(logits, all_targets) # (batch_size,)

    #     return instance_loss, all_preds, all_targets
    
    def inst_eval_out(
            self, 
            att : torch.Tensor, 
            emb : torch.Tensor,
            classifier : torch.nn.Module
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate instance-level loss for out-of-the-class attention branch.

        Arguments:
            att: Attention values of shape `(bag_size,)`.
            emb: Embeddings of shape `(bag_size, feat_dim)`.
            classifier: Instance classifier.
        
        Returns:
            instance_loss: Instance loss.
            p_preds: Predicted instance labels of shape `(k_sample,)`.
            p_targets: Generated instance labels of shape `(k_sample,)`.
        """
        device = att.device
        bag_size = emb.shape[0]

        k_sample = min(self.k_sample, bag_size)

        top_p_ids = torch.topk(att, k_sample)[1] # (k_sample,)
        top_p = torch.index_select(emb, dim=0, index=top_p_ids) # (k_sample, feat_dim)
        p_targets = self.create_negative_targets(k_sample, device) # (k_sample,)
        logits = classifier(top_p) # (k_sample, 2)
        p_preds = torch.topk(logits, 1, dim = 1)[1] # (k_sample,)
        instance_loss = self.inst_loss_fn(logits, p_targets) # (k_sample,)
        return instance_loss, p_preds, p_targets

    # def inst_eval_out(
    #     self,
    #     att : torch.Tensor,
    #     emb : torch.Tensor,
    #     classifier : torch.nn.Module
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Arguments:
    #         att: Attention values of shape `(batch_size, bag_size,)`.
    #         emb: Embeddings of shape `(batch_size, bag_size, feat_dim)`.
    #         classifier: Instance classifier.
        
    #     Returns:
    #         instance_loss: Instance loss.
    #         p_preds: Predicted instance labels of shape `(batch_size, k_sample,)`.
    #         p_targets: Generated instance labels of shape `(batch_size, k_sample,)`.
    #     """

    #     device = att.device
    #     batch_size, bag_size, feat_dim = emb.shape
    #     k_sample = min(self.k_sample, bag_size)

    #     # Get top-k positive indices for the entire batch
    #     top_p_ids = torch.topk(att, k_sample, dim=1)[1] # (batch_size, k_sample)

    #     # Get top-k positive embeddings for the entire batch
    #     top_p = torch.gather(emb, 1, top_p_ids.unsqueeze(-1).expand(-1, -1, feat_dim)) # (batch_size, k_sample, feat_dim)

    #     # Create negative targets for the entire batch
    #     p_targets = self.create_negative_targets(k_sample, device) # (k_sample,)

    #     # Get logits for the entire batch
    #     logits = classifier(top_p) # (batch_size, k_sample, 2)

    #     # Get predicted instance labels for the entire batch
    #     p_preds = torch.topk(logits, 1, dim = 2)[1].squeeze(-1) # (batch_size, k_sample)

    #     # Compute instance loss for the entire batch
    #     instance_loss = self.inst_loss_fn(logits, p_targets) # (batch_size,)

    #     return instance_loss, p_preds, p_targets


    def compute_inst_loss(
        self, 
        att : torch.Tensor,
        emb : torch.Tensor,
        labels : torch.Tensor
    ) -> torch.Tensor:
        """
        Computes instance loss.

        Arguments:
            att: Attention values of shape `(batch_size, bag_size)`.
            emb: Embeddings of shape `(batch_size, bag_size, feat_dim)`.
            labels: Bag labels of shape `(batch_size,)`.
        
        Returns:
            inst_loss: Instance loss.
        """

        sum_inst_loss = 0.0
        batch_size = att.shape[0]

        for i in range(batch_size):
            label = int(labels[i].item())
            if label == 0:
                in_idx = 0
                out_idx = 1
            else:
                in_idx = 1
                out_idx = 0    
            inst_loss_in, _, _ = self.inst_eval(att[i], emb[i], self.inst_classifiers[in_idx])
            inst_loss_out, _, _ = self.inst_eval_out(att[i], emb[i], self.inst_classifiers[out_idx])

            sum_inst_loss += inst_loss_in + inst_loss_out
        return sum_inst_loss


    def forward(
        self, 
        X : torch.Tensor,
        mask : torch.Tensor = None,
        return_att : bool = False,
        return_emb : bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention values (before normalization) in addition to `bag_pred`.
            return_emb: If True, returns embeddings in addition to `bag_pred`.
        
        Returns:
            bag_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
            emb: Only returned when `return_emb=True`. Embeddings of shape (batch_size, bag_size, feat_dim).
        """

        X = self.feat_ext(X) # (batch_size, bag_size, D)

        z, f = self.pool(X, mask, return_att=True) # z: (batch_size, D), f: (batch_size, bag_size)

        bag_pred = self.classifier(z).squeeze(1) # (batch_size,)
        
        if return_emb:
            if return_att:
                return bag_pred, f, X
            else:
                return bag_pred, X
        elif return_att:
            return bag_pred, f
        else:
            return bag_pred
    
    def compute_loss(
        self, 
        labels : torch.Tensor, 
        X : torch.Tensor,
        mask : torch.Tensor,
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
        bag_pred, att, emb = self.forward(X, mask, return_att = True, return_emb=True)
        crit_loss = self.criterion(bag_pred.float(), labels.float())
        crit_name = self.criterion.__class__.__name__
        inst_loss = self.compute_inst_loss(att, emb, labels)

        return bag_pred, { crit_name: crit_loss, 'InstLoss' : inst_loss}

    @torch.no_grad()
    def predict(
        self, 
        X : torch.Tensor, 
        mask : torch.Tensor = None,
        return_inst_pred : bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bag labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.

        Returns:
            bag_pred: Predicted bag labels of shape `(batch_size,)`.
            inst_pred: Predicted instance labels of shape `(batch_size, bag_size)`. Only returned when `return_inst_pred=True`.
        """
        return self.forward(X, mask, return_att=return_inst_pred)
