
# Reimplementation of the dense mincut pooling layer to work with sparse tensors.
# Code from https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.dense.dense_mincut_pool.html#torch_geometric.nn.dense.dense_mincut_pool


import torch


class dense_mincut_pool(torch.nn.Module):
    
    def __init__(self, temp=1.0):
        super(dense_mincut_pool, self).__init__()
        self.temp = temp
    
    def forward(self, x, adj, s, mask=None):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s

        (batch_size, num_nodes, _), k = x.size(), s.size(-1)

        s = torch.softmax(s / self.temp , dim=-1)

        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            x, s = x * mask, s * mask

        out = torch.matmul(s.transpose(1, 2), x)
        # out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
        sT_adj = torch.bmm(adj, s).transpose(1, 2)
        out_adj = torch.matmul(sT_adj, s)

        # MinCut regularization.
        mincut_num = self._rank3_trace(out_adj)
        d_flat = torch.einsum('ijk->ij', adj)
        d = self._rank3_diag(d_flat)
        mincut_den = self._rank3_trace(torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
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

    def _rank3_trace(self, x):
        return torch.einsum('ijj->i', x)

    def _rank3_diag(self, x):
        if x.is_sparse:
            x = x.to_dense()
        eye = torch.eye(x.size(1), device=x.device).type_as(x)
        out = eye * x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1))
        return out