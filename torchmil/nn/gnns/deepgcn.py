import torch
from .gnn_identity import GNNIdentity

class DeepGCNLayer(torch.nn.Module):
    """
    Implementation of a DeepGCN layer.

    Adapts the implementation from [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.DeepGCNLayer.html).
    """
    def __init__(
        self,
        conv : torch.nn.Module = None,
        norm : torch.nn.Module = None,
        act : torch.nn.Module = None,
        block : str = 'plain',
        dropout : float = 0.0
    ):
        """
        Arguments:
            conv: Convolutional layer.
            norm: Normalization layer.
            act: Activation layer.
            block: Skip connection type. Possible values: 'res', 'res+', 'dense', 'plain'.
            dropout: Dropout rate.
        """
        super(DeepGCNLayer, self).__init__()
        self.conv = conv if conv is not None else GNNIdentity()
        self.norm = norm if norm is not None else torch.nn.Identity()
        self.act = act if act is not None else torch.nn.Identity()
        self.dropout = torch.nn.Dropout(dropout)
        self.block = block

    def forward(
        self,
        x : torch.Tensor,
        adj : torch.Tensor
    ) -> torch.Tensor:
        """
        Forward method.

        Arguments:
            x: Node features of shape `(batch_size, n_nodes, in_dim)`.
            adj: Adjacency matrix of shape `(batch_size, n_nodes, n_nodes)`.

        Returns:
            y: Output tensor of shape `(batch_size, n_nodes, out_dim)`.
        """

        if self.block == 'res+':
            y = self.norm(x)
            y = self.act(y)
            y = self.dropout(y)
            y = self.conv(y, adj)
            y = y + x
        else:
            y = self.conv(x, adj)
            y = self.norm(y)
            y = self.act(y)
            if self.block == 'res':
                y = y + x
            elif self.block == 'dense':
                y = torch.cat([x, y], dim=-1)
            y = self.dropout(y)
        return y
