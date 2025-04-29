import torch

class GNNIdentity(torch.nn.Module):
    """
    Identity layer for GNNs.

    This is a placeholder layer that does nothing. It can be used in cases where
    a layer is required but no operation is needed.
    """
    def forward(self, x, adj=None):
        return x