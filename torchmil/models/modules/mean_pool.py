import torch

class MeanPool(torch.nn.Module):
    """
    Mean pooling module.    
    """
    def __init__(self):
        """
        Class constructor
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
            mask = torch.ones(batch_size, bag_size, device=X.device)
        mask = mask.unsqueeze(dim=-1) # (batch_size, bag_size, 1)

        z = torch.sum(X*mask, dim=1) / torch.sum(mask, dim=1) # (batch_size, in_dim)

        return z