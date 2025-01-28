
import os

import numpy as np

import warnings

import torch

from tensordict import TensorDict

from torchmil.utils import build_adj, normalize_adj, add_self_loops

class ProcessedMILDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        features_path: str,
        labels_path: str,
        inst_labels_path: str = None,
        coords_path: str = None,
        dist_thr: float = 1.5,
        norm_adj: bool = True
    ) -> None:
        """
        Class constructor.

        Arguments:
            features_path: Path to the directory containing the features.
            labels_path: Path to the directory containing the bag labels.
            inst_labels_path: Path to the directory containing the instance labels.
            coords_path: Path to the directory containing the coordinates.
            dist_thr: Distance threshold for building the adjacency matrix.
            norm_adj: If True, normalize the adjacency matrix.
        """
        super().__init__()

        self.features_path = features_path
        self.labels_path = labels_path
        self.inst_labels_path = inst_labels_path
        self.coords_path = coords_path
        self.dist_thr = dist_thr
        self.norm_adj = norm_adj

        self.bag_names = [ file for file in os.listdir(self.features_path) if file.endswith('.npy') ]

        self.loaded_bags = {}

    def _load_bag(self, name: str) -> dict[str, np.ndarray]:
        """
        Load a bag from disk.

        Arguments:
            name: Name of the bag to load.

        Returns:
            bag_dict: Dictionary containing the features, label, instance labels and coordinates of the bag.
        """

        bag_dict = {}

        features_file = os.path.join(self.features_path, name)
        if not os.path.exists(features_file):
            warnings.warn(f"Features file {features_file} not found. Setting features to None.")
            bag_dict['features'] = None
        else:
            bag_dict['features'] = np.load(features_file) # (bag_size, ...)

        label_file = os.path.join(self.labels_path, name)
        if not os.path.exists(label_file):
            warnings.warn(f"Label file {label_file} not found. Setting label to None.")
            bag_dict['label'] = None
        else:
            bag_dict['label'] = np.load(label_file) # (1, )

        if self.inst_labels_path is not None:
            inst_labels_file = os.path.join(self.inst_labels_path, name)
            if not os.path.exists(inst_labels_file):
                warnings.warn(f"Instance labels file {inst_labels_file} not found. Setting instance labels to None.")
                bag_dict['inst_labels'] = None
            else:
                bag_dict['inst_labels'] = np.load(inst_labels_file)
                    
        if self.coords_path is not None:
            coords_file = os.path.join(self.coords_path, name)
            if not os.path.exists(coords_file):
                warnings.warn(f"Coordinates file {coords_file} not found. Setting coordinates to None.")
                bag_dict['coords'] = None
            else:
                bag_dict['coords'] = np.load(coords_file) # (bag_size, 2)

        return bag_dict

    def _build_adj(self, bag_dict: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the adjacency matrix of a bag.

        Arguments:
            bag_dict: Dictionary containing the features, label, instance labels and coordinates of the bag.

        Returns:
            edge_index: Edge index of the adjacency matrix.
            edge_weight: Edge weight of the adjacency matrix.
            norm_edge_weight: Normalized edge weight of the adjacency matrix.
        """

        bag_size = bag_dict['coords'].shape[0]
        edge_index, edge_weight = build_adj(
            bag_dict['coords'], bag_dict['features'], dist_thr=self.dist_thr)
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
            
                - features: Features of the bag.
                - label: Label of the bag.
                - inst_labels: Instance labels of the bag.
                - coords: Coordinates of the bag.
                - edge_index: Edge index of the adjacency matrix.
                - edge_weight: Edge weight of the adjacency matrix.
        """

        bag_name = self.bag_names[index]

        if bag_name in self.loaded_bags:
            bag_dict = self.loaded_bags[bag_name]
        else:
            bag_dict = self._load_bag(bag_name)

            if 'coords' in bag_dict:
                edge_index, edge_weight, norm_edge_weight = self._build_adj(bag_dict)
                bag_dict['edge_index'] = edge_index
                if self.norm_adj:
                    bag_dict['edge_weight'] = norm_edge_weight
                else:
                    bag_dict['edge_weight'] = norm_edge_weight
            
            bag_dict['coords'] = (bag_dict['coords'] / self.patch_size).astype(np.int32)

            bag_dict = {k: torch.from_numpy(v) for k,v in bag_dict.items() if k != 'label'}
            if type(bag_dict['label']) == np.ndarray:
                bag_dict['label'] = torch.from_numpy(bag_dict['label'])
            else:
                bag_dict['label'] = torch.as_tensor(bag_dict['label'])

            self.loaded_bags[bag_name] = bag_dict

        return TensorDict(bag_dict)