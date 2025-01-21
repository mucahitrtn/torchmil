
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

            - 'data': Bag features of shape `(bag_size, feat_dim)`.
            - 'label': Bag label.
            - 'inst_labels': Instance labels of shape `(bag_size,)`.
            - 'adj': Adjacency matrix of shape `(bag_size, bag_size)`.
            - 'pos': Instance positions of shape `(bag_size, pos_dim)`.

        sparse: If True, returns sparse adjacency matrices.
    
    Returns:
        batch_dict: Dictionary with the following keys:
        
            - 'data': Padded bag features of shape `(batch_size, max_bag_size, feat_dim)`.
            - 'label': Bag labels of shape `(batch_size,)`.
            - 'inst_labels': Padded instance labels of shape `(batch_size, max_bag_size)`.
            - 'adj': Padded adjacency matrices of shape `(batch_size, max_bag_size, max_bag_size)`.
            - 'pos': Padded instance positions of shape `(batch_size, max_bag_size, pos_dim)`.
            - 'mask': Mask of shape `(batch_size, max_bag_size)`.
    """

    if len(batch_list) == 1:

        batch = batch_list[0]

        data = batch['data'].unsqueeze(0) # (1, bag_size, feat_dim)
        labels = batch['label'].unsqueeze(0) # (1, )
        inst_labels = batch['inst_labels'].unsqueeze(0) # (1, bag_size)
        adj = batch['adj'].unsqueeze(0) # (1, bag_size, bag_size)
        if adj.is_sparse:
            if not sparse:
                adj = adj.to_dense()
            else:
                adj = adj.coalesce()
        pos = batch['pos'].unsqueeze(0) # (1, bag_size, pos_dim)
        mask = torch.ones_like(inst_labels).float() # (1, bag_size)
        batch_dict = {
            'data': data,
            'label': labels,
            'inst_labels': inst_labels,
            'adj': adj,
            'pos': pos,
            'mask': mask
        }
    else:

        batch_dict = {}
        for key in batch_list[0].keys():
            batch_dict[key] = [batch[key] for batch in batch_list]
        
        data, mask = pad_tensors(batch_dict['data']) # (batch_size, max_bag_size, feat_dim), (batch_size, max_bag_size)
        batch_dict['data'] = data
        batch_dict['mask'] = mask
        if 'inst_labels' in batch_dict:
            inst_labels, _ = pad_tensors(batch_dict['inst_labels']) # (batch_size, max_bag_size)
            batch_dict['inst_labels'] = inst_labels
        if 'pos' in batch_dict:
            pos, _ = pad_tensors(batch_dict['pos'])
            batch_dict['pos'] = pos
        batch_dict['label'] = torch.stack(batch_dict['label']) # (batch_size, )

        if 'adj' in batch_dict:
            edge_index_list = []
            edge_val_list = []
            adj_shape_list = []

            for adj in batch_dict['adj']:
                edge_index_list.append(adj.indices())
                edge_val_list.append(adj.values())
                adj_shape_list.append(adj.shape)

            adj_shape_array = np.array(adj_shape_list) # (batch_size, 2)
            adj_max_shape = tuple(np.max(adj_shape_array, axis=0).astype(int))
            adj_list = []
            for i in range(len(batch_list)):
                indices = edge_index_list[i]
                values = edge_val_list[i]
                adj_list.append(torch.sparse_coo_tensor(indices, values, adj_max_shape))
            adj = torch.stack(adj_list).coalesce() # (batch_size, bag_size, bag_size)
            if not sparse:
                adj = adj.to_dense()
            
            batch_dict['adj'] = adj
        
    return TensorDict(batch_dict)