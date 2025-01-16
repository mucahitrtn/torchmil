import torch

def masked_softmax(X, mask, dim=-1):
    """
    Compute masked softmax.
    
    Arguments:
        X (Tensor): Input tensor of shape `(batch_size, bag_size)`.
        mask (Tensor): Mask of shape `(batch_size, bag_size)`.
    
    Returns:
        Tensor: Masked softmax of shape `(batch_size, bag_size)`.
    """

    exp_X = torch.exp(X)
    masked_exp_X = exp_X * mask
    sum_masked_exp_X = masked_exp_X.sum(dim=dim, keepdim=True)
    return masked_exp_X / sum_masked_exp_X

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