import torch

from torchmil.nn import GCNConv, DeepGCNLayer

from torchmil.nn.utils import get_feat_dim, LazyLinear, masked_softmax

class DeepGraphSurv(torch.nn.Module):
    def __init__(self, 
        in_shape: tuple = None, 
        n_layers_rep : int = 4,
        n_layers_att : int = 2,
        hidden_dim : int = None,
        att_dim : int = 128,
        dropout : float = 0.0,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        criterion=torch.nn.BCEWithLogitsLoss()
    ):
        """
        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension). If not provided, it will be lazily initialized.
            n_layers_rep: Number of GCN layers in the representation network.
            n_layers_att: Number of GCN layers in the attention network.
            hidden_dim: Hidden dimension. If not provided, it will be set to the feature dimension.
            att_dim: Attention dimension.
            dropout: Dropout rate.
            feat_ext: Feature extractor.
            criterion: Loss function.       
        """
        super(DeepGraphSurv, self).__init__()
        self.criterion = criterion
        self.feat_ext = feat_ext

        if in_shape is not None:
            feat_dim = get_feat_dim(feat_ext, in_shape)
        else:
            feat_dim = None
        
        if hidden_dim is None:
            hidden_dim = feat_dim

        self.layers_rep = torch.nn.ModuleList()
        for i in range(n_layers_rep):
            conv_layer = GCNConv( feat_dim if i == 0 else hidden_dim, hidden_dim, add_self_loops=True)
            norm_layer = torch.nn.LayerNorm(hidden_dim, elementwise_affine=True)
            act_layer = torch.nn.ReLU()
            self.layers_rep.append(
                DeepGCNLayer(conv_layer, norm_layer, act_layer, dropout=dropout, block='plain')
            )
        
        self.layers_att = torch.nn.ModuleList()
        for i in range(n_layers_att):
            conv_layer = GCNConv(hidden_dim if i==0 else att_dim, att_dim, add_self_loops=True)
            norm_layer = torch.nn.LayerNorm(att_dim, elementwise_affine=True)
            act_layer = torch.nn.ReLU()
            self.layers_att.append(
                DeepGCNLayer(conv_layer, norm_layer, act_layer, dropout=dropout, block='plain')
            )
        
        self.proj1d = LazyLinear(att_dim, 1)

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
        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)

        for layer in self.layers_rep:
            X = layer(X, adj)
        
        H = X # (batch_size, bag_size, hidden_dim)
        for layer in self.layers_att:
            H = layer(H, adj) # (batch_size, bag_size, att_dim)
        f = self.proj1d(H) # (batch_size, bag_size, 1)
        s = masked_softmax(f, mask) # (batch_size, bag_size, 1)
        
        z = torch.bmm(X.transpose(1,2), s).squeeze(-1) # (batch_size, hidden_dim)
        
        Y_pred = self.classifier(z).squeeze(-1) # (batch_size,)

        if return_att:
            return Y_pred, f.squeeze(-1)
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