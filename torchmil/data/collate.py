
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

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
    Collate function for MIL datasets.

    Arguments:
        batch_list: List of dictionaries with the following keys:

            - 'features': Bag features of shape `(bag_size, feat_dim)`.
            - 'label': Bag label.
            - 'inst_labels': Instance labels of shape `(bag_size,)`.
            - 'edge_index': Edge index of the adjacency matrix.
            - 'edge_weight': Edge weight of the adjacency matrix.
            - 'coords': Instance positions of shape `(bag_size, pos_dim)`.

        sparse: If True, returns sparse adjacency matrices.
    
    Returns:
        batch_dict: Dictionary with the following keys:
        
            - 'features': Padded bag features of shape `(batch_size, max_bag_size, feat_dim)`.
            - 'label': Bag labels of shape `(batch_size,)`.
            - 'inst_labels': Padded instance labels of shape `(batch_size, max_bag_size)`.
            - 'adj': Padded adjacency matrices of shape `(batch_size, max_bag_size, max_bag_size)`.
            - 'coords': Padded instance positions of shape `(batch_size, max_bag_size, pos_dim)`.
            - 'mask': Mask of shape `(batch_size, max_bag_size)`.
    """

    batch_dict = {}
    key_list = batch_list[0].keys()
    for key in key_list:
        batch_dict[key] = [bag_dict[key] for bag_dict in batch_list]
    
    features, mask = pad_tensors(batch_dict['features']) # (batch_size, max_bag_size, feat_dim), (batch_size, max_bag_size)
    batch_dict['features'] = features
    batch_dict['mask'] = mask
    if 'inst_labels' in batch_dict:
        inst_labels, _ = pad_tensors(batch_dict['inst_labels']) # (batch_size, max_bag_size)
        batch_dict['inst_labels'] = inst_labels
    if 'coords' in batch_dict:
        pos, _ = pad_tensors(batch_dict['coords'])
        batch_dict['coords'] = pos
    batch_dict['label'] = torch.stack(batch_dict['label']) # (batch_size, )

    if 'edge_index' in batch_dict:
        edge_index_list = batch_dict['edge_index']
        edge_weight_list = batch_dict['edge_weight']

        bag_size_list = [len(batch_list[i]['features']) for i in range(len(batch_list))]
        max_bag_size = max(bag_size_list)

        adj_list = []
        for i in range(len(edge_index_list)):
            edge_index = edge_index_list[i]
            edge_weight = edge_weight_list[i]
            adj_list.append(torch.sparse_coo_tensor(edge_index, edge_weight, (max_bag_size, max_bag_size)))
        
        adj = torch.stack(adj_list).coalesce() # (batch_size, bag_size, bag_size)
        if not sparse:
            adj = adj.to_dense()
        
        batch_dict['adj'] = adj

        # Remove the edge index and edge weight from the batch_dict
        del batch_dict['edge_index']
        del batch_dict['edge_weight']
    
    return TensorDict(batch_dict)