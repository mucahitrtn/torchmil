import torch
import numpy as np

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
    mask : torch.Tensor = None,
    ) -> torch.Tensor:
    """
    Compute masked softmax along the second dimension.
    
    Arguments:
        X (Tensor): Input tensor of shape `(batch_size, N, ...)`.
        mask (Tensor): Mask of shape `(batch_size, N)`. If None, no masking is applied.
    
    Returns:
        Tensor: Masked softmax of shape `(batch_size, N, ...)`.
    """

    if mask is None:
        return torch.nn.functional.softmax(X, dim=1)

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

def get_spatial_representation(
        X : torch.Tensor,
        coords : torch.Tensor,
    ) -> torch.Tensor:
    """
    Computes the spatial representation of a bag given the sequential representation and the coordinates.

    Given the input tensor `X` of shape `(batch_size, bag_size, dim)` and the coordinates `coords` of shape `(batch_size, bag_size, n)`, 
    this function returns the spatial representation `X_enc` of shape `(batch_size, coord1, coord2, ..., coordn, dim)`.

    This representation is characterized by the fact that the coordinates are used to index the elements of spatial representation:
    `X_enc[batch, i1, i2, ..., in, :] = X[batch, idx, :]` where `(i1, i2, ..., in) = coords[batch, idx]`.

    Arguments:
        X (Tensor): Sequential representation of shape `(batch_size, bag_size, dim)`.
        coords (Tensor): Coordinates of shape `(batch_size, bag_size, n)`.
    
    Returns:
        X_esp: Spatial representation of shape `(batch_size, coord1, coord2, ..., coordn, dim)`.
    """

    # Get the shape of the spatial representation
    batch_size = X.shape[0]
    bag_size = X.shape[1]
    n = coords.shape[-1]
    shape = torch.Size([batch_size] + [int(coords[:, :, i].max().item()) + 1 for i in range(n)] + [X.shape[-1]])

    # Initialize the spatial representation
    X_enc = torch.zeros(shape, device=X.device, dtype=X.dtype)

    # Create batch indices of shape (batch_size, bag_size)
    batch_indices = torch.arange(batch_size, device=X.device).unsqueeze(1).expand(-1, bag_size)

    # Create a list of spatial indices (one per coordinate dimension), each of shape (batch_size, bag_size)
    spatial_indices = [coords[:, :, i] for i in range(n)]

    # Build the index tuple without using the unpack operator in the subscript.
    index_tuple = (batch_indices,) + tuple(spatial_indices)

    # Use advanced indexing to assign values from X into X_enc.
    X_enc[index_tuple] = X


    return X_enc

def get_sequential_representation(
        X_esp : torch.Tensor,
        coords : torch.Tensor,
    ) -> torch.Tensor:
    """
    Computes the sequential representation of a bag given the spatial representation and the coordinates.

    Given the spatial tensor `X_esp` of shape `(batch_size, coord1, coord2, ..., coordn, dim)` and the coordinates `coords` of shape `(batch_size, bag_size, n)`, 
    this function returns the sequential representation `X` of shape `(batch_size, bag_size, dim)`.

    This representation is characterized by the fact that the coordinates are used to index the elements of spatial representation:
    `X_seq[batch, idx, :] = X_esp[batch, i1, i2, ..., in, :]` where `(i1, i2, ..., in) = coords[batch, idx]`.

    Arguments:
        X_esp (Tensor): Spatial representation of shape `(batch_size, coord1, coord2, ..., coordn, dim)`.
        coords (Tensor): Coordinates of shape `(batch_size, bag_size, n)`.
    
    Returns:
        X_seq: Sequential representation of shape `(batch_size, bag_size, dim)`.
    """

    batch_size = X_esp.shape[0]
    bag_size = coords.shape[1]
    n = coords.shape[-1]

    # Create batch indices with shape (batch_size, bag_size)
    batch_indices = torch.arange(batch_size, device=X_esp.device).unsqueeze(1).expand(-1, bag_size)

    # Build the index tuple without using the unpack operator in the subscript.
    # Each element in the tuple has shape (batch_size, bag_size)
    index_tuple = (batch_indices,) + tuple(coords[:, :, i] for i in range(n))

    # Use advanced indexing to extract the sequential representation from X_esp.
    # The result will have shape (batch_size, bag_size, dim)
    X_seq = X_esp[index_tuple]

    return X_seq







class SinusoidalPositionalEncodingND(torch.nn.Module):
    def __init__(self, n_dim, channels, dtype_override=None):
        """
        Positional encoding for tensors of arbitrary dimensions.

        Arguments:
            n_dim (int): Number of dimensions.
            channels (int): Number of channels.
            dtype_override (torch.dtype): Data type override.
        """
        super(SinusoidalPositionalEncodingND, self).__init__()
        self.n_dim = n_dim
        self.org_channels = channels
        channels = int(np.ceil(channels / (2*n_dim)) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.dtype_override = dtype_override
        self.channels = channels

        print("Channels: ", channels)
    
    def _get_embedding(self, sin_inp):
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, tensor):
        """
        Arguments:
            tensor (Tensor): Input tensor of shape `(batch_size, l1, l2, ..., lN, channels)`.
        
        Returns:
            Tensor: Positional encoding of shape `(batch_size, l1, l2, ..., lN, channels)`.
        """
        if len(tensor.shape) != self.n_dim + 2:
            raise RuntimeError("The input tensor has to be {}d!".format(self.n_dim + 2))

        shape = tensor.shape

        orig_ch = shape[-1]
        emb_shape = list(shape)[1:]
        emb_shape[-1] = self.channels * self.n_dim

        emb = torch.zeros(
            emb_shape,
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )

        for i in range(self.n_dim):
            pos = torch.arange(shape[i+1], device=tensor.device, dtype=self.inv_freq.dtype)
            sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
            emb_i = self._get_embedding(sin_inp)
            for _ in range(self.n_dim-i-1):
                emb_i = emb_i.unsqueeze(1)
            emb[..., i*self.channels : (i+1)*self.channels] = emb_i

        return emb[None, ..., :orig_ch].repeat(shape[0], *(1 for _ in range(self.n_dim)), 1)