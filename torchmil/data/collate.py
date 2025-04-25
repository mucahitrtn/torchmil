import torch
from torch import Tensor

from tensordict import TensorDict

import numpy as np

def pad_tensors(
        tensor_list : list[Tensor],
        padding_value : int = 0
    ) -> tuple[Tensor, Tensor]:
    """
    Pads a list of tensors to the same shape and returns a mask.

    Arguments:
        tensor_list: List of tensors, each of shape `(bag_size, ...)`.
        padding_value: Value to pad with.

    Returns:
        padded_tensor: Padded tensor of shape `(batch_size, max_bag_size, ...)`.
        mask: Mask of shape `(batch_size, max_bag_size)`.
    """

    if len(tensor_list) == 1:
        padded_tensor = tensor_list[0].unsqueeze(0) # (1, bag_size, ...)
        mask = torch.ones((1, tensor_list[0].size(0)), dtype=torch.uint8, device=tensor_list[0].device) # (1, bag_size)
    else:
        # Determine the maximum bag size
        max_bag_size = max(tensor.size(0) for tensor in tensor_list)
        feature_shape = tensor_list[0].size()[1:]

        batch_size = len(tensor_list)
        padded_tensor = torch.full((batch_size, max_bag_size, *feature_shape), padding_value, dtype=tensor_list[0].dtype, device=tensor_list[0].device)
        mask = torch.zeros((batch_size, max_bag_size), dtype=torch.uint8, device=tensor_list[0].device)

        for i, tensor in enumerate(tensor_list):
            bag_size = tensor.size(0)
            padded_tensor[i, :bag_size] = tensor
            mask[i, :bag_size] = 1

    return padded_tensor, mask

def collate_fn(
        batch_list : list[dict[str, torch.Tensor]],
        sparse : bool = True,
    ) -> TensorDict:
    """
    Collate function for MIL datasets. Given a list of bags (represented as dictionaries)
    it pads the tensors in the bag to the same shape. Then, it returns a dictionary representing
    the batch. The keys in the dictionary are the keys in the bag dictionaries. Additionally,
    the returned dictionary contains a mask for the padded tensors. This mask is 1 where the
    tensor is not padded and 0 where the tensor is padded.

    Arguments:
        batch_list: List of dictionaries. Each dictionary represents a bag and should contain the same keys. The values can be:

            - Tensors of shape `(bag_size, ...)`. In this case, the tensors are padded to the same shape.
            - Sparse tensors in COO format. In this case, the resulting sparse tensor has shape `(batch_size, max_bag_size, max_bag_size)`, where `max_bag_size` is the maximum bag size in the batch. If `sparse=False`, the sparse tensor is converted to a dense tensor.

        sparse: If True, the sparse tensors are returned as sparse tensors. If False, the sparse tensors are converted to dense tensors.

    Returns:
        batch_dict: Dictionary with the same keys as the bag dictionaries. The values are tensors of shape `(batch_size, max_bag_size, ...)` or sparse tensors of shape `(batch_size, max_bag_size, max_bag_size)`. Additionally, the dictionary contains a mask of shape `(batch_size, max_bag_size)`.
    """

    batch_dict = {}
    key_list = batch_list[0].keys()
    for key in key_list:
        batch_dict[key] = [bag_dict[key] for bag_dict in batch_list]

    for key in key_list:
        data_list = batch_dict[key]
        if not data_list[0].is_sparse:
            if data_list[0].dim() == 0:
                data = torch.stack(data_list)
            else:
                data, mask = pad_tensors(data_list)
                batch_dict[key] = data
                if 'mask' not in batch_dict:
                    batch_dict['mask'] = mask
        else:
            index_list = []
            value_list = []
            size_list = []
            for i in range(len(data_list)):
                index_list.append(data_list[i].coalesce().indices().numpy())
                value_list.append(data_list[i].coalesce().values().numpy())
                size_list.append(data_list[i].size())
            max_size = max(size_list)

            data_list = [ torch.sparse_coo_tensor(index, value, max_size) for index, value in zip(index_list, value_list) ]
            batch_dict[key] = torch.stack(data_list).coalesce()
            if not sparse:
                batch_dict[key] = batch_dict[key].to_dense()

    return TensorDict(batch_dict)
