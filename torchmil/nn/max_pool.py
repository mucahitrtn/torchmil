import torch

class MaxPool(torch.nn.Module):
    r"""
    Max pooling aggregation.

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times D}$,
    this model aggregates the instance features into a bag representation $\mathbf{z} \in \mathbb{R}^{D}$ as,

    $$
    \left[ \mathbf{z} \right]_d = \max \left\{ \left[ \mathbf{x}_n \right]_{d} \ \colon n \in \left\{ 1, \ldots, N \right\}  \right\},
    $$

    where $\left[ \mathbf{a} \right]_i$ denotes the $i$-th element of the vector $\mathbf{a}$.
    """
    def __init__(self):
        """
        """
        super(MaxPool, self).__init__()

    def forward(
        self,
        X : torch.Tensor,
        mask : torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.
            mask: Mask tensor of shape `(batch_size, bag_size)`.

        Returns:
            z: Output tensor of shape `(batch_size, in_dim)`.
        """

        batch_size, bag_size, _ = X.shape

        if mask is None:
            mask = torch.ones(batch_size, bag_size, device=X.device).bool()
        mask = mask.unsqueeze(dim=-1)

        # Set masked values to -inf
        X = X.masked_fill(~mask, float('-inf'))
        z = X.max(dim=1)[0]

        return z
