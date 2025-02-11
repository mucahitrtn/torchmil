import torch

class LazyLinear(torch.nn.Module):
    def __init__(self, in_features=None, out_features=512, bias=True, device=None, dtype=None):
        super().__init__()

        if in_features is not None:
            self.module = torch.nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        else:
            self.module = torch.nn.LazyLinear(out_features, bias=bias, device=device, dtype=dtype)
    
    def forward(self, x):
        return self.module(x)

def masked_softmax(
    X : torch.Tensor,
    mask : torch.Tensor,
    ) -> torch.Tensor:
    """
    Compute masked softmax.
    
    Arguments:
        X (Tensor): Input tensor of shape `(batch_size, N, ...)`.
        mask (Tensor): Mask of shape `(batch_size, N)`.
    
    Returns:
        Tensor: Masked softmax of shape `(batch_size, N, ...)`.
    """

    while mask.dim() < X.dim():
        mask = mask.unsqueeze(-1)
        
    X_masked = X.masked_fill(mask == 0, -float('inf'))

    return torch.nn.functional.softmax(X_masked, dim=1)

class MaskedSoftmax(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, X, mask):
        """
        Compute masked softmax.
        
        Arguments:
            X (Tensor): Input tensor of shape `(batch_size, bag_size)`.
            mask (Tensor): Mask of shape `(batch_size, bag_size)`.
        
        Returns:
            Tensor: Masked softmax of shape `(batch_size, bag_size)`.
        """

        return masked_softmax(X, mask, dim=self.dim)
    
def get_feat_dim(
        feat_ext : torch.nn.Module,
        input_shape : tuple[int, ...]
    ) -> int:
    """
    Get feature dimension of the feature extractor.

    Arguments:
        feat_ext (torch.nn.Module): Feature extractor.
        input_shape (tuple): Input shape of the feature extractor.
    """
    with torch.no_grad():
        return feat_ext(torch.zeros((1, *input_shape))).shape[-1]