
import os

import numpy as np

import warnings

import torch
from torch import Tensor

from tensordict import TensorDict

from torchmil.utils import build_adj_WSI, normalize_adj, add_self_loops

class WSIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        labels_path: str,
        inst_labels_path: str = None,
        coords_path: str = None,
        patch_size: int = 512,
    ) -> None:
        """
        Class constructor.

        Arguments:
            data_path: Path to the directory containing the data.
            labels_path: Path to the directory containing the bag labels.
            inst_labels_path: Path to the directory containing the instance labels.
            coords_path: Path to the directory containing the coordinates.
        """
        super().__init__()

        self.data_path = data_path
        self.labels_path = labels_path
        self.inst_labels_path = inst_labels_path
        self.coords_path = coords_path
        self.patch_size = patch_size

        self.bag_names = [ file for file in os.listdir(self.data_path) if file.endswith('.npy') ]
        # self.bag_names = [ file.split('.')[0] for file in self.bag_names ]

        self.loaded_bags = {}

    def _load_bag(self, name: str) -> dict[str, np.ndarray]:
        """
        Load a bag from disk.

        Arguments:
            name: Name of the bag to load.

        Returns:
            bag_dict: Dictionary containing the data, label, instance labels and coordinates of the bag.
        """

        bag_dict = {}

        bag_dict['data'] = np.load(os.path.join(self.data_path, name))  # (bag_size, ...)
        bag_dict['label'] = np.load(os.path.join(self.labels_path, name))  # (1, )

        if self.inst_labels_path is not None:
            file_path = os.path.join(self.inst_labels_path, name)
            if os.path.exists(file_path):
                bag_dict['inst_labels'] = np.load(os.path.join(
                        self.inst_labels_path, name)) # (bag_size, )
            else:
                if bag_dict['label'][0] == 0:
                    bag_dict['inst_labels'] = np.zeros(bag_dict['data'].shape[0])
                else:
                    warnings.warn(
                        f'Instance labels not found for {name}. Setting all to -1.')
                    bag_dict['inst_labels'] = np.full(bag_dict['data'].shape[0], -1)
                    
        if self.coords_path is not None:
            bag_dict['coords'] = np.load(os.path.join(self.coords_path, name))  # (bag_size, 2)

        return bag_dict

    def _build_adj(self, bag_dict: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the adjacency matrix of a bag.

        Arguments:
            bag_dict: Dictionary containing the data, label, instance labels and coordinates of the bag.

        Returns:
            edge_index: Edge index of the adjacency matrix.
            edge_weight: Edge weight of the adjacency matrix.
            norm_edge_weight: Normalized edge weight of the adjacency matrix.
        """

        bag_size = bag_dict['coords'].shape[0]
        edge_index, edge_weight = build_adj_WSI(
            bag_dict['coords'], bag_dict['data'], patch_size=self.patch_size)
        norm_edge_weight = normalize_adj(
            edge_index, edge_weight, n_nodes=bag_size)
        if bag_size == 1:
            edge_index, norm_edge_weight = add_self_loops(edge_index, bag_size, norm_edge_weight)

        return edge_index, edge_weight, norm_edge_weight

    def __len__(self) -> int:
        """
        Returns:
            Number of bags in the dataset
        """
        return len(self.bag_names)

    def __getitem__(self, index: int) -> TensorDict:
        """
        Arguments:
            index: Index of the bag to retrieve.
        
        Returns:
            bag_dict: Dictionary containing the following keys:
            
                - data: Data of the bag.
                - label: Label of the bag.
                - inst_labels: Instance labels of the bag.
                - coords: Coordinates of the bag.
                - edge_index: Edge index of the adjacency matrix.
                - edge_weight: Edge weight of the adjacency matrix.
                - norm_edge_weight: Normalized edge weight of the adjacency matrix.        
        """

        bag_name = self.bag_names[index]

        if bag_name in self.loaded_bags:
            bag_dict = self.loaded_bags[bag_name]
        else:
            bag_dict = self._load_bag(bag_name)

            if 'coords' in bag_dict:
                edge_index, edge_weight, norm_edge_weight = self._build_adj(bag_dict)
                bag_dict['edge_index'] = edge_index
                bag_dict['edge_weight'] = edge_weight
                bag_dict['norm_edge_weight'] = norm_edge_weight
            
            bag_dict['coords'] = (bag_dict['coords'] / self.patch_size).astype(np.int32)

            bag_dict = {k: torch.as_tensor(v) for k, v in bag_dict.items()}

            self.loaded_bags[bag_name] = bag_dict

        return TensorDict(bag_dict)