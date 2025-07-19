import torch

from torchmil.nn.gnns.chebconv import ChebConv


# Helper function to create a batched sparse COO tensor
def create_batched_sparse_adj(batch_size, n_nodes, density=0.1, device="cpu"):
    """
    Creates a batched sparse COO adjacency matrix (torch.sparse_coo_tensor).
    The indices will have 3 rows: [batch_idx, row_idx, col_idx].
    """
    num_edges_per_graph = int(n_nodes * n_nodes * density)

    all_indices = []
    all_values = []

    for b in range(batch_size):
        # Ensure at least one edge to avoid empty sparse tensor issues for sum
        if num_edges_per_graph == 0 and n_nodes > 0:
            rows = torch.tensor([0], device=device)
            cols = torch.tensor([0], device=device)
            num_edges_graph = 1
        else:
            rows = torch.randint(0, n_nodes, (num_edges_per_graph,), device=device)
            cols = torch.randint(0, n_nodes, (num_edges_per_graph,), device=device)
            num_edges_graph = num_edges_per_graph

        # Add batch index for this graph
        batch_idx_tensor = torch.full(
            (num_edges_graph,), b, dtype=torch.long, device=device
        )

        # Stack indices for this graph (batch_idx, row_idx, col_idx)
        graph_indices = torch.stack([batch_idx_tensor, rows, cols], dim=0)
        all_indices.append(graph_indices)
        all_values.append(
            torch.ones(num_edges_graph, dtype=torch.float32, device=device)
        )

    # Concatenate all indices and values across the batch
    if not all_indices or sum(v.numel() for v in all_values) == 0:
        indices = torch.empty((3, 0), dtype=torch.long, device=device)
        values = torch.empty((0,), dtype=torch.float32, device=device)
    else:
        indices = torch.cat(all_indices, dim=1)
        values = torch.cat(all_values, dim=0)

    # Create the sparse tensor with the logical shape (batch_size, n_nodes, n_nodes)
    adj = torch.sparse_coo_tensor(
        indices, values, (batch_size, n_nodes, n_nodes), dtype=torch.float32
    )
    return adj


# Helper function to create a batched dense adjacency matrix
def create_batched_dense_adj(batch_size, n_nodes, density=0.1, device="cpu"):
    """
    Creates a batched dense adjacency matrix.
    """
    adj = torch.rand(batch_size, n_nodes, n_nodes, device=device)
    adj = (adj > (1 - density)).float()  # Make it binary for adjacency concept
    return adj


class TestChebConv:
    def test_initialization(self):
        """Test ChebConv initialization."""
        in_channels = 16
        out_channels = 32
        K = 3
        conv = ChebConv(in_channels, out_channels, K=K)
        assert conv.in_channels == in_channels
        assert conv.out_channels == out_channels
        assert conv.K == K
        assert conv.compute_lambda_max is True
        assert isinstance(conv.fc, torch.nn.Linear)
        assert conv.fc.in_features == K * in_channels
        assert conv.fc.out_features == out_channels

        conv_no_lambda_comp = ChebConv(
            in_channels, out_channels, compute_lambda_max=False
        )
        assert conv_no_lambda_comp.compute_lambda_max is False

    def test_forward_pass_with_dense_adj_compute_lambda_max(self):
        """
        Test forward pass with dense adj and compute_lambda_max=True.
        This should now pass.
        """
        in_channels = 8
        out_channels = 16
        K = 3
        batch_size = 2
        n_nodes = 5

        conv = ChebConv(in_channels, out_channels, K=K, compute_lambda_max=True)

        x = torch.randn(batch_size, n_nodes, in_channels, dtype=torch.float32)
        adj = create_batched_dense_adj(batch_size, n_nodes, density=0.5)

        out = conv.forward(x, adj)

        assert out.shape == (batch_size, n_nodes, out_channels)
        assert torch.isfinite(out).all()  # Check for NaNs or Infs

    def test_forward_pass_with_dense_adj_fixed_lambda_max(self):
        """
        Test forward pass with dense adj and a provided lambda_max.
        This should now pass.
        """
        in_channels = 8
        out_channels = 16
        K = 3
        batch_size = 2
        n_nodes = 5

        conv = ChebConv(in_channels, out_channels, K=K, compute_lambda_max=False)

        x = torch.randn(batch_size, n_nodes, in_channels, dtype=torch.float32)
        adj = create_batched_dense_adj(batch_size, n_nodes, density=0.5)

        # Provide a dummy lambda_max (e.g., 2.0 for each graph in batch)
        lambda_max_val = (
            torch.tensor([2.0] * batch_size, dtype=torch.float32, device=x.device)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )  # (batch_size, 1, 1)

        out = conv.forward(x, adj, lambda_max=lambda_max_val)

        assert out.shape == (batch_size, n_nodes, out_channels)
        assert torch.isfinite(out).all()

    def test_forward_pass_with_sparse_adj_compute_lambda_max(self):
        """
        Test forward pass with sparse adj and compute_lambda_max=True.
        """
        in_channels = 8
        out_channels = 16
        K = 3
        batch_size = 2
        n_nodes = 5

        conv = ChebConv(in_channels, out_channels, K=K, compute_lambda_max=True)

        x = torch.randn(batch_size, n_nodes, in_channels, dtype=torch.float32)
        adj = create_batched_sparse_adj(batch_size, n_nodes, density=0.2)

        out = conv.forward(x, adj)

        assert out.shape == (batch_size, n_nodes, out_channels)
        assert torch.isfinite(out).all()

    def test_forward_pass_with_sparse_adj_fixed_lambda_max(self):
        """
        Test forward pass with sparse adj and a provided lambda_max.
        """
        in_channels = 8
        out_channels = 16
        K = 3
        batch_size = 2
        n_nodes = 5

        # compute_lambda_max doesn't matter here as lambda_max is provided
        conv = ChebConv(in_channels, out_channels, K=K, compute_lambda_max=False)

        x = torch.randn(batch_size, n_nodes, in_channels, dtype=torch.float32)
        adj = create_batched_sparse_adj(batch_size, n_nodes, density=0.2)

        # Provide a dummy lambda_max (e.g., 2.0 for each graph in batch)
        lambda_max_val = (
            torch.tensor([2.0] * batch_size, dtype=torch.float32, device=x.device)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        out = conv.forward(x, adj, lambda_max=lambda_max_val)

        assert out.shape == (batch_size, n_nodes, out_channels)
        assert torch.isfinite(out).all()
