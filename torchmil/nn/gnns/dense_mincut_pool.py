import torch

from torchmil.nn.utils import masked_softmax

def trace(x : torch.Tensor) -> torch.Tensor:
    """
    Trace of a rank-3 tensor.

    Arguments:
        x: Input tensor of shape `(batch_size, n, n)`.

    Returns:
        Tensor: Trace of the input tensor of shape `(batch_size,)`.
    """
    return torch.einsum('bnn->b', x)

def diag(x : torch.Tensor) -> torch.Tensor:
    """
    Given a rank-2 tensor, return a rank-3 tensor with the input tensor as the diagonal.

    Arguments:
        x: Input tensor of shape `(batch_size, n)`.

    Returns:
        Tensor: Rank-3 tensor of shape `(batch_size, n, n)
    """
    if x.is_sparse:
        x = x.to_dense()
    eye = torch.eye(x.size(1), device=x.device).type_as(x)
    out = eye * x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1))
    return out

def dense_mincut_pool(
    x : torch.Tensor,
    adj : torch.Tensor,
    s : torch.Tensor,
    mask : torch.Tensor = None,
    temp : float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dense MinCut Pooling.

    Adapts the implementation from [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.dense.dense_mincut_pool.html).

    Arguments:
        x: Input tensor of shape `(batch_size, n_nodes, in_dim)`.
        adj: Adjacency tensor of shape `(batch_size, n_nodes, n_nodes)`.
        s: Dense learned assignments tensor of shape `(batch_size, n_nodes, n_cluster)`.
        mask: Mask tensor of shape `(batch_size, n_nodes)`.
        temp: Temperature.

    Returns:
        x_: Pooled node feature tensor of shape `(batch_size, n_cluster, in_dim)`.
        adj_: Coarsened adjacency tensor of shape `(batch_size, n_cluster, n_cluster)`.
        mincut_loss: MinCut loss.
        ortho_loss: Orthogonality loss.
    """
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _) = x.size()
    k = s.size(-1)

    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    adj_s = torch.bmm(adj, s)
    out_adj = torch.matmul(s.transpose(1, 2), adj_s)
    
    # out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # MinCut regularization.
    mincut_num = trace(out_adj)
    d_flat = torch.einsum('ijk->ij', adj)
    d = diag(d_flat)
    mincut_den = trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    EPS = 1e-15

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, mincut_loss, ortho_loss
