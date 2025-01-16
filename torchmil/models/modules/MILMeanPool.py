import torch

class MILMeanPool(torch.nn.Module):
    def __init__(self, in_dim, **kwargs):
        super(MILMeanPool, self).__init__()
        self.in_dim = in_dim

    def forward(self, X, mask=None, return_attval=False, **kwargs):
        """
        input:
            X: tensor (batch_size, bag_size, in_dim)
            adj_mat: sparse coo tensor (batch_size, bag_size, bag_size)
            mask: tensor (batch_size, bag_size)
        output:
            z: tensor (batch_size, in_dim)
        """
        batch_size = X.shape[0]
        bag_size = X.shape[1]
        in_dim = X.shape[2]

        if mask is None:
            mask = torch.ones(batch_size, bag_size, device=X.device)
        mask = mask.unsqueeze(dim=-1) # (batch_size, bag_size, 1)

        z = torch.sum(X*mask, dim=1) / torch.sum(mask, dim=1) # (batch_size, in_dim)

        if return_attval:
            return z, None
        else:
            return z