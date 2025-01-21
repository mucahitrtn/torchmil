import torch

class MaxPool(torch.nn.Module):
    """
    Max pooling module.    
    """
    def __init__(self):
        """
        Class constructor.        
        """
        super(MaxPool, self).__init__()
    
    def forward(
        self,
        X : torch.Tensor,
        mask : torch.Tensor = None,
    ) -> torch.Tensor:
        """
        input:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.
            mask: Mask tensor of shape `(batch_size, bag_size)`.
        output:
            z: Output tensor of shape `(batch_size, in_dim)`.
        """
        
        batch_size, bag_size, _ = X.shape

        if mask is None:
            mask = torch.ones(batch_size, bag_size, device=X.device)
        mask = mask.unsqueeze(dim=-1)

        # Set masked values to -inf
        X = X.masked_fill(~mask, float('-inf'))
        z = X.max(dim=1)[0]

        return z