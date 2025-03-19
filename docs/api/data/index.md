# torchmil.data

## Data representation in <tt><b>torchmil</b></tt>

!!! note
    The data representation in <tt><b>torchmil</b></tt> is inspired by the data representation in [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs).

    See [this notebook](../../examples/data_representation.ipynb) for a detailed explanation of the data representation in <tt><b>torchmil</b></tt>.

In Multiple Instance Learning (MIL), a bag is a collection of instances. 
In <tt><b>torchmil</b></tt>, a bag is represented as a [TensorDict](https://pytorch.org/tensordict/stable/index.html).
In most cases (e.g., the datasets provided in <tt><b>torchmil</b></tt>), a bag  will contain at least two keys:

- `bag['X']`: a tensor of shape `(bag_size, ...)` containing the instances in the bag. Usually, this tensor is called _bag feature matrix_, since these instances are feature vectors extracted from the raw representation of the instances, and therefore it has shape `(bag_size, feature_dim)`. 
- `bag['Y']`: a tensor containing the label of the bag. In the simplest case, this tensor is a scalar, but it can be a tensor of any shape (e.g., in multi-class MIL).

Additionally, a bag may contain other keys. The most common ones in <tt><b>torchmil</b></tt> are:

- `bag['y_inst']`: a tensor of shape `(bag_size, ...)` containing the labels of the instances in the bag. In the pure MIL setting, this tensor is only used for evaluation purposes since the label of the instances are not known. However, some methods may require some sort of supervision at the instance level.
- `bag['adj']`: a tensor of shape `(bag_size, bag_size)` containing the adjacency matrix of the bag. This matrix is used to represent the relationships between the instances in the bag. The methods implemented in [<tt><b>torchmil.models</b></tt>](../models/index.md) allow this matrix to be a sparse tensor.
- `bag['coords']`: a tensor of shape `(bag_size, coords_dim)` containing the coordinates of the instances in the bag. This tensor is used to represent the absolute position of the instances in the bag.

## Batches in <tt><b>torchmil</b></tt>

Batching is crucial for training deep learning models. However, in MIL, each bag can be of different size. To solve this, in <tt><b>torchmil</b></tt>, the tensors in the bags are padded to the maximum size of the bags in the batch. A mask tensor is used to indicate which elements of the padded tensors are real instances and which are padding. This mask tensor is used to adjust the behavior of the models to ignore the padding elements (e.g., in the attention mechanism). 

The function `torchmil.data.collate_fn` is used to collate a list of bags into a batch. This function can be used as the `collate_fn` argument of the PyTorch `DataLoader`. The function `torchmil.data.pad_tensors` is used to pad the tensors in the bags.

-------------------------
::: torchmil.data.collate_fn
-------------------------
::: torchmil.data.pad_tensors