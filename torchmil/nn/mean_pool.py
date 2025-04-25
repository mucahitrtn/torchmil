import torch

class MeanPool(torch.nn.Module):
    r"""
    Mean pooling aggregation.

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times D}$,
    this model aggregates the instance features into a bag representation $\mathbf{z} \in \mathbb{R}^{D}$ as,

    $$
        \mathbf{z} = \frac{1}{N} \sum_{n=1}^{N} \mathbf{x}_n.
    $$
    """
    def __init__(self):
        """
        """
        super(MeanPool, self).__init__()

    def forward(
        self,
        X : torch.Tensor,
        mask : torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.
            mask: Mask tensor of shape `(batch_size, bag_size)`.
        Returns:
            z: Output tensor of shape `(batch_size, in_dim)`.
        """
        batch_size, bag_size, _ = X.shape

        if mask is None:
            mask = torch.ones(batch_size, bag_size, device=X.device).bool()
        mask = mask.unsqueeze(dim=-1) # (batch_size, bag_size, 1)

        eff_num_inst = torch.sum(mask, dim=1) # (batch_size, 1)

        if torch.any(eff_num_inst == 0):
            z = torch.zeros(batch_size, X.shape[-1], device=X.device)
        else:
            z = torch.sum(X*mask, dim=1) / torch.sum(mask, dim=1) # (batch_size, in_dim)

        return z
