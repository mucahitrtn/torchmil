
# Notes

### Data Handling of Bags. 

A single bag in `torchmil` is described by an instance of `torchmil.Bag`. 
This class is used to store all the information about a bag. It is similar to the `torch_geometric.data.Data` in Pytorch Geometric. It has the following attributes:

- `bag.X`: The features of the bag (a 2D tensor of shape (num_instances, num_features)).
- `bag.edge_index`: The edge index of the bag (a 2D tensor of shape (2, num_edges)).
- `bag.edge_val`: The edge values of the bag (a 1D tensor of shape (num_edges)).
- `bag.edge_attr`: The edge attributes of the bag (a 2D tensor of shape (num_edges, num_edge_features)).
- `bag.Y`: The label of the bag (a 1D tensor of shape (1)).
- `bag.y`: The label of the instances in the bag (a 1D tensor of shape (num_instances)).
- `bag.pos`: The position of the instances in the bag (a 2D tensor of shape (num_instances, num_position_features)).

### Documentation style

We will use the [Google style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for the documentation of the code.