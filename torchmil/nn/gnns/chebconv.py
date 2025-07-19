import torch


class ChebConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 3,
        compute_lambda_max: bool = True,
    ) -> None:
        """
        Chebyshev spectral graph convolutional operator tailored for sparse adjacency matrices (COO format).
        Proposed in [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375).
        This class follows closely the design of [PyTorch Geometric's ChebConv](https://pytorch-geometric.readthedocs.io/en/2.5.0/generated/torch_geometric.nn.conv.ChebConv.html), but adapted for COO format adjacency matrices.

        Arguments:
            in_channels: Number of input channels (features per node).
            out_channels: Number of output channels (features per node).
            K: Order of the Chebyshev polynomial approximation. Default is 3.
            compute_lambda_max: If True, computes the maximum eigenvalue of the adjacency matrix for normalization. If False and not provided during forward pass, it will be set it to 2.0.
        """
        super(ChebConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.compute_lambda_max = compute_lambda_max

        self.fc = torch.nn.Linear(K * in_channels, out_channels, bias=False)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, lambda_max: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of the Chebyshev convolutional layer.

        Arguments:
            x: Input tensor of shape `(batch_size, n_nodes, in_channels)`.
            adj: Adjacency matrix of shape `(batch_size, n_nodes, n_nodes)`. It can be a sparse tensor (in COO format) or a dense tensor.
            lambda_max: Optional maximum eigenvalue of the adjacency matrix for normalization. If not provided, it will be computed.

        Returns:
            out: Output tensor of shape `(batch_size, n_nodes, out_channels)`.
        """

        # degree = torch.sparse.sum(adj, dim=1).to_dense().unsqueeze(-1)  # Shape: (batch_size, n_nodes, 1)
        if adj.is_sparse:
            degree = (
                torch.sparse.sum(adj, dim=1).to_dense().unsqueeze(-1)
            )  # (batch_size, n_nodes, 1)
        else:
            degree = adj.sum(dim=1).unsqueeze(-1)  # Shape: (batch_size, n_nodes, 1)

        if lambda_max is None:
            if self.compute_lambda_max:
                lambda_max_list = []
                batch_size = x.shape[0]
                for i in range(batch_size):
                    eig, _ = torch.linalg.eig(adj[i].to_dense())

                    lambda_max_list.append(eig.real.max().item())
                lambda_max = (
                    torch.tensor(lambda_max_list, device=x.device)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )  # Shape: (batch_size, 1, 1)
            else:
                lambda_max = 2.0

        lambda_ratio = 2.0 / (lambda_max + 1e-6)

        Z_list = []
        Z_list.append(x)
        Z_1 = lambda_ratio * degree * x - lambda_ratio * torch.bmm(adj, x) - x
        Z_list.append(Z_1)
        for k in range(2, self.K):
            Z_k = (
                2
                * (
                    lambda_ratio * degree * Z_list[k - 1]
                    - lambda_ratio * torch.bmm(adj, Z_list[k - 1])
                    - Z_list[k - 1]
                )
                - Z_list[k - 2]
            )
            Z_list.append(Z_k)

        Z = torch.stack(Z_list, dim=1)  # Shape: (batch_size, K, n_nodes, in_channels)
        Z = Z.view(
            Z.shape[0], Z.shape[2], -1
        )  # Shape: (batch_size, n_nodes, K * in_channels)
        out = self.fc(Z)  # Shape: (batch_size, n_nodes, out_channels)

        return out
