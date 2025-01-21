

import torch
import numpy as np

from copy import deepcopy
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence

from tensordict import TensorDict


# from torchmil.data import Bag

# class MILDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         bag_names: list[str],
#         data_dict: dict[str, np.ndarray],
#         labels_dict: dict[str, int],
#         inst_labels_dict: dict[str, np.ndarray],
#         pos_dict: dict[str, np.ndarray],
#     ):
#         """
#         Arguments:
#             bag_names: List of bag names.
#             data_dict: Dictionary with bag data. Keys are bag names, values are numpy arrays of shape `(bag_size, ...)`.
#             labels_dict: Dictionary with bag labels. Keys are bag names, values are integers.
#             inst_labels_dict: Dictionary with instance labels. Keys are bag names, values are numpy arrays of shape `(bag_size,)`.
#             pos_dict: Dictionary with instance positions. Keys are bag names, values are numpy arrays of shape `(bag_size, 2)`.
#         """
#         self.bag_names = bag_names
#         self.data_dict = data_dict
#         self.labels_dict = labels_dict
#         self.inst_labels_dict = inst_labels_dict
#         self.pos_dict = pos_dict

#         self.bag_list = []

#         for key in self.bag_names:
#             assert key in self.data_dict
#             assert key in self.labels_dict
#             assert key in self.inst_labels_dict
#             assert key in self.pos_dict

#             assert len(self.data_dict[key]) == len(self.inst_labels_dict[key])
#             assert len(self.data_dict[key]) == len(self.pos_dict[key])

#     def __len__(self):
#         return len(self.bag_names)

#     def __getitem__(self, idx):
#         bag_name = self.bag_names[idx]

#         bag = Bag(
#             name=bag_name,
#             data=self.data_dict[bag_name],
#             label=self.labels_dict[bag_name],
#             inst_labels=self.inst_labels_dict[bag_name],
#             pos=self.pos_dict[bag_name],
#         )

#         return bag

class MILDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root: str,
        bag_names: list[str],
    ) -> None:
        """
        `root` is the path to the directory containing the bags. It is expected that this directory contains:
        - data: Directory containing the bag data.
        - labels: Directory containing the bag labels.
        - inst_labels: Directory containing the instance labels.
        - adj: Directory containing the adjacency matrices.
        - pos: Directory containing the instance positions.
        """
        self.root = root
        self.bag_names = bag_names

    def __len__(self) -> int:
        return len(self.bag_names)
    
    def _load_bag_data(self, bag_name: str) -> np.ndarray:
        return np.load(f'{self.root}/data/{bag_name}.npy')
    
    def _load_bag_label(self, bag_name: str) -> int:
        return np.load(f'{self.root}/labels/{bag_name}.npy')
    
    def _load_inst_labels(self, bag_name: str) -> np.ndarray:
        return np.load(f'{self.root}/inst_labels/{bag_name}.npy')
    
    def _load_adj(self, bag_name: str) -> np.ndarray:
        return np.load(f'{self.root}/adj/{bag_name}.npy')
    
    def _load_pos(self, bag_name: str) -> np.ndarray:
        return np.load(f'{self.root}/pos/{bag_name}.npy')

    def __getitem__(self, index):
        bag_name = self.bag_names[index]
        bag = TensorDict(
            name=bag_name,
            data=self._load_bag_data(bag_name),
            label=self._load_bag_label(bag_name),
            inst_labels=self._load_inst_labels(bag_name),
            adj=self._load_adj(bag_name),
            pos=self._load_pos(bag_name),
        )
        return bag
            
        

