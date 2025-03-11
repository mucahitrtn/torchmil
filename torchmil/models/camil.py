import torch
import torch.nn as nn

from torchmil.nn import NystromTransformerLayer
from torchmil.models import MILModel

from torchmil.nn.utils import masked_softmax, get_feat_dim


class CAMILSelfAttention(nn.Module):
    def __init__(
        self, 
        in_dim : int,
        att_dim : int = 512
    ) -> None:
        super(CAMILSelfAttention, self).__init__()
        self.qk_nn = torch.nn.Linear(in_dim, 2*att_dim, bias = False)
        self.v_nn = torch.nn.Linear(in_dim, in_dim, bias = False)

    def forward(
        self, 
        X : torch.Tensor,
        adj : torch.Tensor
    ) -> torch.Tensor:
        """
        Arguments:
            X: (batch_size, bag_size, in_dim)
            adj: (batch_size, bag_size, bag_size)
        
        Returns:
            L: (batch_size, bag_size, in_dim)
        """
        
        q, k = self.qk_nn(X).chunk(2, dim=-1) # (batch_size, bag_size, att_dim), (batch_size, bag_size, att_dim)
        v = self.v_nn(X) # (batch_size, bag_size, in_dim)
        att_dim = q.shape[-1]

        inv_scale = 1.0 / (att_dim**0.5)

        if adj.is_sparse:
            adj = adj.to_dense()

        w = inv_scale*(q.matmul(k.transpose(-2, -1)) * adj).sum(dim = -1, keepdim = True) # (batch_size, bag_size, 1)
        
        L = w.softmax(dim=1)*v # (batch_size, bag_size, in_dim)

        return L

class CAMILAttentionPool(nn.Module):
    def __init__(
        self, 
        in_dim : int, 
        att_dim : int = 128
    ) -> None:
        super(CAMILAttentionPool, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, att_dim)
        self.fc2 = torch.nn.Linear(att_dim, 1, bias=False)

    def forward(
        self, 
        t : torch.Tensor,
        m : torch.Tensor,
        mask : torch.Tensor,
        return_att : bool = False
    ) -> torch.Tensor:
        """
        Arguments:
            t: (batch_size, bag_size, in_dim)
            m: (batch_size, bag_size, in_dim)
            mask: (batch_size, bag_size)
            return_att: If True, returns attention values in addition to `z`.
        
        Returns:
            z: (batch_size, in_dim)
            f: (batch_size, bag_size) if `return_att
        """

        f = self.fc2(torch.nn.functional.tanh(self.fc1(t))) # (batch_size, bag_size, 1)
        a = masked_softmax(f, mask) # (batch_size, bag_size, 1)
        z = torch.bmm(m.transpose(1,2), a).squeeze(dim=2) # (batch_size, in_dim)

        if return_att:
            return z, f.squeeze(dim=2)
        else:
            return z

class CAMIL(MILModel):
    r"""
    Method proposed in the paper [CAMIL: Context-Aware Multiple Instance Learning for Cancer Detection and Subtyping in Whole Slide Images](https://arxiv.org/abs/2305.05314).

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$, 
    this model optionally transforms the instance features using a feature extractor trained with self-supervised contrastive learning:

    $$ \mathbf{X} = \text{FeatExt}(\mathbf{X}) \in \mathbb{R}^{N \times D}.$$

    Then, it applies a Transformer module to capture global dependencies among instances:

    $$ \mathbf{H} = \text{Transformer}(\mathbf{X}) \in \mathbb{R}^{N \times D}.$$

    To incorporate local context, the model introduces a neighbor-constrained attention mechanism that recalibrates instance-level attention scores based on spatially adjacent instances:

    $$ \mathbf{A} = \text{NeighborAttention}(\mathbf{H}, \mathbf{Adj}) \in \mathbb{R}^{N},$$

    where $\mathbf{Adj}$ represents a similarity-based adjacency matrix encoding spatial relationships.

    Finally, a feature aggregation step adaptively fuses local and global representations to predict the bag label:

    $$ \mathbf{z} = \sum_{i=1}^{N} a_i \mathbf{m}_i,$$

    where $\mathbf{m}_i$ is a blended local-global representation, and $a_i$ are attention weights.

    The final classification is obtained via a linear layer applied to $\mathbf{z}$.
"""

    def __init__(
        self,
        in_shape: tuple,
        nystrom_att_dim : int = 512,
        pool_att_dim : int = 128,
        n_heads : int = 4,
        n_landmarks : int = None,
        pinv_iterations : int = 6,
        residual : bool = True,
        dropout : float = 0.0,
        use_mlp : bool = False,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        criterion : torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        r"""
        Arguments:
            in_shape: Shape of the input data expected by the feature extractor (excluding batch dimension).
            nystrom_att_dim: Dimensionality of the embedding when computing self-attention.
            pool_att_dim: Dimensionality of the embedding in the attention pooling.
            n_heads: Number of attention heads in the Nyströmformer layer.
            n_landmarks: Number of landmarks in the Nyströmformer layer.
            pinv_iterations: Number of iterations for the pseudo-inverse computation in the Nyströmformer layer.
            residual: Whether to use residual connections in the Nyströmformer layer.
            dropout: Dropout rate in the Nyströmformer layer.
            use_mlp: Whether to use a MLP after the Nyströmformer layer.
            feat_ext: Feature extractor. By default, the identity function (no feature extraction).
            criterion: Loss function used for training (default: Binary Cross-Entropy loss from logits).
"""

        super(CAMIL, self).__init__()
        self.feat_ext = feat_ext
        self.criterion = criterion

        feat_dim = get_feat_dim(feat_ext, in_shape)

        if feat_dim != nystrom_att_dim:
            self.fc1 = torch.nn.Linear(feat_dim, nystrom_att_dim)
        else:
            self.fc1 = torch.nn.Identity()

        self.nystrom_transformer_layer = NystromTransformerLayer(att_dim=nystrom_att_dim, n_heads=n_heads, n_landmarks=n_landmarks, pinv_iterations=pinv_iterations, residual=residual, dropout=dropout, use_mlp=use_mlp)
        
        self.camil_self_attention = CAMILSelfAttention(in_dim=feat_dim, att_dim=nystrom_att_dim)
        self.camil_att_pool = CAMILAttentionPool(in_dim=feat_dim, att_dim=pool_att_dim)

        self.classifier = nn.Linear(feat_dim, 1)

        self.criterion = criterion
    
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

        X = self.feat_ext(X) # (batch_size, bag_size, feat_dim)
        T = self.nystrom_transformer_layer(self.fc1(X)) # (batch_size, bag_size, feat_dim)

        L = self.camil_self_attention(T, adj) # (batch_size, bag_size, feat_dim)

        M = torch.sigmoid(L)*L + (1 - torch.sigmoid(L))*T # (batch_size, bag_size, feat_dim)

        if return_att:
            z, att = self.camil_att_pool(T, M, mask, return_att=True) # (batch_size, feat_dim), (batch_size, bag_size)
        else:
            z = self.camil_att_pool(T, M, mask) # (batch_size, feat_dim)

        Y_pred = self.classifier(z).squeeze(dim=-1) # (batch_size,)
        
        if return_att:
            return Y_pred, att
        else:
            return Y_pred
    
    def compute_loss(
        self,
        Y: torch.Tensor,
        X: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value.
        """

        Y_pred = self.forward(X, adj, mask, return_att=False)

        crit_loss = self.criterion(Y_pred.float(), Y.float())
        crit_name = self.criterion.__class__.__name__

        return Y_pred, {crit_name: crit_loss}
    
    def predict(
        self,
        X: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
        return_inst_pred: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_inst_pred (bool): If `True`, returns instance labels predictions, in addition to bag label predictions.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            y_inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        return self.forward(X, adj, mask, return_att=return_inst_pred)