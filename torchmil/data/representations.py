import torch

def seq_to_spatial(
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

def spatial_to_seq(
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
