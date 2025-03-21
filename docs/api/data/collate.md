!!! note
    See [this notebook](../../examples/data_representation.ipynb) for an explanation with examples of how batching is performed in <tt><b>torchmil</b></tt>.

# Batches in <tt><b>torchmil</b></tt>

Batching is crucial for training deep learning models. However, in MIL, each bag can be of different size. To solve this, in <tt><b>torchmil</b></tt>, the tensors in the bags are padded to the maximum size of the bags in the batch. A mask tensor is used to indicate which elements of the padded tensors are real instances and which are padding. This mask tensor is used to adjust the behavior of the models to ignore the padding elements (e.g., in the attention mechanism). 

!!! question "Why not use [`torch.nested`](https://pytorch.org/docs/stable/nested.html)?"
    `torch.nested` offer a more flexible method for handling bags of varying sizes. However, since the PyTorch API for nested tensors is still in the prototype stage, <tt><b>torchmil</b></tt> currently relies on the padding approach.

The function `torchmil.data.collate_fn` is used to collate a list of bags into a batch. This function can be used as the `collate_fn` argument of the PyTorch `DataLoader`. The function `torchmil.data.pad_tensors` is used to pad the tensors in the bags.

-------------------------
::: torchmil.data.collate_fn
-------------------------
::: torchmil.data.pad_tensors