import torch

from torchmil.nn import LazyLinear

class GCNConv(torch.nn.Module):
    """
    Implementation of a Graph Convolutional Network (GCN) layer.

    Adapts the implementation from [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html).
    """
    def __init__(
        self, 
        in_dim : int, 
        out_dim : int,
        add_self_loops : bool = False,
        bias : bool = True
        ):
        """
        Arguments:
            in_dim: Input dimension.
            out_dim: Output dimension.
            add_self_loops: Whether to add self-loops.
            bias: Whether to use bias.        
        """

        super(GCNConv,self).__init__()

        self.fc = LazyLinear(in_dim, out_dim, bias=bias)
        self.add_self_loops = add_self_loops

    def forward(
        self,
        x : torch.Tensor,
        adj : torch.Tensor
    ) -> torch.Tensor:
        """        
        Arguments:
            x : Node features of shape (batch_size, n_nodes, in_dim).
            adj : Adjacency matrix of shape (batch_size, n_nodes, n_nodes).
        
        Returns:
            y : Output tensor of shape (batch_size, n_nodes, out_dim).        
        """

        y = torch.bmm(adj, x)
        if self.add_self_loops:
            y += x
        y = self.fc(y)

        return y