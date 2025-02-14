import torch

from torchmil.nn import AttentionPool, GCNConv, DeepGCNLayer

from torchmil.nn.utils import get_feat_dim, LazyLinear

class PathGCN(torch.nn.Module):
    def __init__(self, 
        in_shape : tuple = None,
        n_gcn_layers : int = 4,
        mlp_depth : int = 1,
        hidden_dim : int = None,
        att_dim : int = 128,
        dropout : float = 0.0,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        criterion=torch.nn.BCEWithLogitsLoss()
    ):
        """
        
        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension). If not provided, it will be lazily initialized.
            n_gcn_layers: Number of GCN layers.
            mlp_depth: Number of layers in the MLP (applied after GCN layers).
            hidden_dim: Hidden dimension. If not provided, it will be set to the feature dimension.
            att_dim: Attention dimension.
            dropout: Dropout rate.
            feat_ext: Feature extractor.
            criterion: Loss function.
        """
        super(PathGCN, self).__init__()
        self.criterion = criterion
        self.feat_ext = feat_ext

        if in_shape is not None:
            feat_dim = get_feat_dim(feat_ext, in_shape)
        else:
            feat_dim = None
        
        if hidden_dim is None:
            hidden_dim = feat_dim

        self.gcn_layers = torch.nn.ModuleList()
        for i in range(n_gcn_layers):
            # conv_layer = GENConv( feat_dim if i == 0 else hidden_dim, hidden_dim, aggr='softmax')
            conv_layer = GCNConv( feat_dim if i == 0 else hidden_dim, hidden_dim, add_self_loops=True)
            norm_layer = torch.nn.LayerNorm(hidden_dim, elementwise_affine=True)
            act_layer = torch.nn.ReLU()
            self.gcn_layers.append(
                DeepGCNLayer(conv_layer, norm_layer, act_layer, dropout=dropout, block='plain')
            )

        self.mlp = torch.nn.ModuleList()
        for _ in range(mlp_depth):
            fc_layer = LazyLinear(hidden_dim, hidden_dim)
            act_layer = torch.nn.ReLU()
            dropout_layer = torch.nn.Dropout(dropout)
            self.mlp.append(
                torch.nn.Sequential(
                    fc_layer,
                    act_layer,
                    dropout_layer
                )
            )

        self.pool = AttentionPool(in_dim = hidden_dim, att_dim = att_dim)
        self.classifier = LazyLinear(hidden_dim, 1)    

    def forward(
        self, 
        X : torch.Tensor,
        adj : torch.Tensor,
        mask : torch.Tensor,
        return_att : bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention values (before normalization) in addition to `Y_pred`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
        """
        
        X = self.feat_ext(X) # (batch_size, bag_size, hidden_dim)
        X_ = X 
        for layer in self.gcn_layers:
            X = layer(X, adj) # (batch_size, bag_size, hidden_dim)
            X_ = torch.cat([X_, X], axis=1) # (batch_size, bag_size*(i+2), hidden_dim)
        
        for layer in self.mlp:
            X_ = layer(X_)
        
        if mask is not None:
            # amplify the masked values
            num_new_nodes = X_.shape[1] - X.shape[1]
            mask = torch.cat([mask, torch.ones(mask.shape[0], num_new_nodes, device=mask.device)], axis=1)
        
        if return_att:
            z, att = self.pool(X_, mask, return_att=True) # (batch_size, hidden_dim)
        else:
            z = self.pool(X_, mask) # (batch_size, hidden_dim)
        
        Y_pred = self.classifier(z).squeeze(1) # (batch_size,)

        if return_att:
            bag_size = X.shape[1]
            att = att[:, :bag_size]
            return Y_pred, att
        else:
            return Y_pred        

    def compute_loss(
        self,
        Y : torch.Tensor,
        X : torch.Tensor,
        adj : torch.Tensor,
        mask : torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`.
        
        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss
        """
        Y_pred = self.forward(X, adj, mask)
        crit_loss = self.criterion(Y_pred.float(), Y.float())
        crit_name = self.criterion.__class__.__name__
        return Y_pred, { crit_name: crit_loss }

    def predict(
        self,
        X : torch.Tensor,
        adj : torch.Tensor,
        mask : torch.Tensor,
        return_inst_pred : bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_inst_pred: If True, returns instance predictions.
        
        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            y_inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        return self.forward(X, adj, mask, return_att=return_inst_pred)