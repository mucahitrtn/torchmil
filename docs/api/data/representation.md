!!! note
    See [this notebook](../../examples/data_representation.ipynb) for an explanation with examples of the different types of representations in <tt><b>torchmil</b></tt>.

# Spatial and sequential representation

In <tt><b>torchmil</b></tt>, bags can be represented in two ways: sequential and spatial. 

In the sequential representation `bag['X']` is a tensor of shape `(bag_size, dim)`.
This representation is the most common in MIL. 

When the bag has some spatial structure, the sequential representation can be coupled with a graph using an adjacency matrix or with the coordinates of the instances. These are stored as `bag['adj']` (of shape `(bag_size, bag_size)`) and `bag['coords']` (of shape `(bag_size, coords_dim)`), respectively.

Alternatively, the spatial representation can be used. 
In this case, `bag['X']` is a tensor of shape `(coord1, ..., coordN, dim)`, where `N=coords_dim` is the number of dimensions of the space.

In <tt>torchmil</tt>, you can convert from one representation to the other using the functions `torchmil.utils.seq_to_spatial` and `torchmil.utils.spatial_to_seq` from the [<tt><b>torchmil.data</b></tt>](./index.md) module. These functions need the coordinates of the instances in the bag, stored as `bag['coords']`.

!!! info "Example: Whole Slide Images"
    Due to their large resolution, Whole Slide Images (WSIs) are usually represented as bags of patches. Each patch is an image, from which a feature vector of is typically extracted. The spatial representation of a WSI has shape `(height, width, feat_dim)`, while the sequential representation has shape `(bag_size, feat_dim)`. The coordinates corresponds to the coordinates of the patches in the WSI. 

    [SETMIL](../models/setmil.md) is an example of a model that uses the spatial representation of a WSI. 

-------------------------
::: torchmil.data.seq_to_spatial
-------------------------
::: torchmil.data.spatial_to_seq