import os
import warnings
import torch
import copy
import numpy as np

from tensordict import TensorDict
from torchmil.utils import build_adj, normalize_adj, add_self_loops

class ProcessedMILDataset(torch.utils.data.Dataset):
    r"""
    This class represents a general MIL dataset where the bags have been processed.

    **MIL processing and directory structure.**
    It is assumed that the bags have been processed and saved as numpy files.
    A feature file should yield an array of shape `(bag_size, ...)`, where `...` represents the shape of the features.
    A label file should yield an array of shape arbitrary shape, e.g., `(1,)` for binary classification.
    An instance label file should yield an array of shape `(bag_size, ...)`, where `...` represents the shape of the instance labels.
    A coordinates file should yield an array of shape `(bag_size, coords_dim)`, where `coords_dim` is the dimension of the coordinates.

    This dataset expects the following directory structure:

    ```
    features_path
    ├── bag1.npy
    ├── bag2.npy
    └── ...
    labels_path
    ├── bag1.npy
    ├── bag2.npy
    └── ...
    inst_labels_path
    ├── bag1.npy
    ├── bag2.npy
    └── ...
    coords_path
    ├── bag1.npy
    ├── bag2.npy
    └── ...
    ```

    **Adjacency matrix.**
    If the coordinates of the instances are available, the adjacency matrix will be built using the Euclidean distance between the coordinates.
    Formally, the adjacency matrix $\mathbf{A} = \left[ A_{ij} \right]$ is defined as:

    \begin{equation}
    A_{ij} = \begin{cases}
    d_{ij}, & \text{if } \left\| \mathbf{c}_i - \mathbf{c}_j \right\| \leq \text{dist_thr}, \\
    0, & \text{otherwise},
    \end{cases} \quad d_{ij} = \begin{cases}
    1, & \text{if } \text{adj_with_dist=False}, \\
    \exp\left( -\frac{\left\| \mathbf{x}_i - \mathbf{x}_j \right\|}{d} \right), & \text{if } \text{adj_with_dist=True}.
    \end{cases}
    \end{equation}

    where $\mathbf{c}_i$ and $\mathbf{c}_j$ are the coordinates of the instances $i$ and $j$, respectively, $\text{dist_thr}$ is a threshold distance,
    and $\mathbf{x}_i \in \mathbb{R}^d$ and $\mathbf{x}_j \in \mathbb{R}^d$ are the features of instances $i$ and $j$, respectively.
    """


    def __init__(
        self,
        features_path: str,
        labels_path: str,
        inst_labels_path: str = None,
        coords_path: str = None,
        bag_names: list = None,
        bag_keys: list = ["X", "Y", "y_inst", "adj", "coords"],
        dist_thr: float = 1.5,
        adj_with_dist: bool = False,
        norm_adj: bool = True,
        load_at_init: bool = True
    ) -> None:
        """
        Class constructor.

        Arguments:
            features_path: Path to the directory containing the features.
            labels_path: Path to the directory containing the bag labels.
            inst_labels_path: Path to the directory containing the instance labels.
            coords_path: Path to the directory containing the coordinates.
            bag_keys: List of keys to use for the bag dictionary. See the `__getitem__` method for more details.
            bag_names: List of bag names to load. If None, all bags are loaded.
            dist_thr: Distance threshold for building the adjacency matrix.
            adj_with_dist: If True, the adjacency matrix is built using the Euclidean distance between the instance features. If False, the adjacency matrix is binary.
            norm_adj: If True, normalize the adjacency matrix.
            load_at_init: If True, load the bags at initialization. If False, load the bags on demand.
        """
        super().__init__()

        self.features_path = features_path
        self.labels_path = labels_path
        self.inst_labels_path = inst_labels_path
        self.coords_path = coords_path
        self.bag_names = bag_names
        self.bag_keys = bag_keys
        self.dist_thr = dist_thr
        self.adj_with_dist = adj_with_dist
        self.norm_adj = norm_adj
        self.load_at_init = load_at_init

        if self.bag_names is None:
            self.bag_names = [ file for file in os.listdir(self.features_path) if file.endswith('.npy') ]
            self.bag_names = [ os.path.splitext(file)[0] for file in self.bag_names ]
        self.bag_names = sorted(self.bag_names)

        self.loaded_bags = {}
        if self.load_at_init:
            for name in self.bag_names:
                self.loaded_bags[name] = self._build_bag(name)

    def _load_features(self, name: str) -> np.ndarray:
        """
        Load the features of a bag from disk.

        Arguments:
            name: Name of the bag to load.

        Returns:
            features: Features of the bag.
        """
        features_file = os.path.join(self.features_path, name+'.npy')
        if not os.path.exists(features_file):
            warnings.warn(f"Features file {features_file} not found. Setting features to None.")
            return None
        return np.load(features_file)

    def _load_labels(self, name: str) -> np.ndarray:
        """
        Load the label of a bag from disk.

        Arguments:
            name: Name of the bag to load.

        Returns:
            label: Label of the bag.
        """
        label_file = os.path.join(self.labels_path, name+'.npy')
        if not os.path.exists(label_file):
            warnings.warn(f"Label file {label_file} not found. Setting label to None.")
            return None
        out = np.load(label_file)
        return out

    def _load_inst_labels(self, name: str) -> np.ndarray:
        """
        Load the instance labels of a bag from disk.

        Arguments:
            name: Name of the bag to load.

        Returns:
            inst_labels: Instance labels of the bag.
        """
        if self.inst_labels_path is None:
            return None

        inst_labels_file = os.path.join(self.inst_labels_path, name+'.npy')
        if not os.path.exists(inst_labels_file):
            warnings.warn(f"Instance labels file {inst_labels_file} not found. Setting instance labels to None.")
            return None
        out = np.load(inst_labels_file)
        return out

    def _load_coords(self, name: str) -> np.ndarray:
        """
        Load the coordinates of a bag from disk.

        Arguments:
            name: Name of the bag to load.

        Returns:
            coords: Coordinates of the bag.
        """
        if self.coords_path is None:
            return None

        coords_file = os.path.join(self.coords_path, name+'.npy')
        if not os.path.exists(coords_file):
            warnings.warn(f"Coordinates file {coords_file} not found. Setting coordinates to None.")
            return None
        return np.load(coords_file)

    def _load_bag(self, name: str) -> dict[str, torch.Tensor]:
        """
        Load a bag from disk.

        Arguments:
            name: Name of the bag to load.

        Returns:
            bag_dict: Dictionary containing the features ('X'), label ('Y'), instance labels ('y_inst') and coordinates ('coords') of the bag.
        """

        bag_dict = {}
        if "X" in self.bag_keys:
            bag_dict['X'] = self._load_features(name)
        
        if "Y" in self.bag_keys:
            bag_dict['Y'] = self._load_labels(name)
        
        if "y_inst" in self.bag_keys:
            bag_dict['y_inst'] = self._load_inst_labels(name)

        if "coords" in self.bag_keys or "adj" in self.bag_keys:
            bag_dict['coords'] = self._load_coords(name)

        return bag_dict

    def _build_bag(self, name: str) -> dict[str, torch.Tensor]:
        """
        Build a bag from the features, labels, instance labels and coordinates. First, it loads the bag from disk using `_load_bag`, then it builds the adjacency matrix using `_build_adj`.

        Arguments:
            name: Name of the bag to build.

        Returns:
            bag_dict: Dictionary containing the features ('X'), label ('Y'), instance labels ('y_inst') and coordinates ('coords') of the bag.
        """
        bag_dict = self._load_bag(name)

        if "adj" in self.bag_keys and bag_dict["coords"] is not None:
            edge_index, edge_weight, norm_edge_weight = self._build_adj(bag_dict)
            if self.norm_adj:
                edge_val = norm_edge_weight
            else:
                edge_val = edge_weight

            bag_dict['adj'] = torch.sparse_coo_tensor(
                edge_index, edge_val, (bag_dict['coords'].shape[0], bag_dict['coords'].shape[0])).coalesce()
            bag_dict['coords'] = torch.from_numpy(bag_dict['coords'])
        
        for key in ["X", "Y", "y_inst"]:
            if key in bag_dict:
                bag_dict[key] = torch.from_numpy(bag_dict[key])

        return bag_dict

    def _build_adj(self, bag_dict: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the adjacency matrix of a bag.

        Arguments:
            bag_dict: Dictionary containing the features ('X'), label ('Y'), instance labels ('y_inst') and coordinates ('coords') of the bag.

        Returns:
            edge_index: Edge index of the adjacency matrix, of shape `(2, n_edges)`.
            edge_weight: Edge weight of the adjacency matrix, of shape `(n_edges,)`.
            norm_edge_weight: Normalized edge weight of the adjacency matrix, of shape `(n_edges,)`.
        """

        bag_size = bag_dict['coords'].shape[0]
        if self.adj_with_dist:
            edge_index, edge_weight = build_adj(
                bag_dict['coords'], bag_dict['X'], dist_thr=self.dist_thr)
        else:
            edge_index, edge_weight = build_adj(
                bag_dict['coords'], None, dist_thr=self.dist_thr)
        norm_edge_weight = normalize_adj(
            edge_index, edge_weight, n_nodes=bag_size)
        if bag_size == 1:
            edge_index, norm_edge_weight = add_self_loops(edge_index, norm_edge_weight, bag_size)

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
            bag_dict: Dictionary containing the keys defined in `bag_keys` and their corresponding values.

                - X: Features of the bag, of shape `(bag_size, ...)`.
                - Y: Label of the bag.
                - y_inst: Instance labels of the bag, of shape `(bag_size, ...)`.
                - adj: Adjacency matrix of the bag. It is a sparse COO tensor of shape `(bag_size, bag_size)`. If `norm_adj=True`, the adjacency matrix is normalized.
                - coords: Coordinates of the bag, of shape `(bag_size, coords_dim)`.
        """

        bag_name = self.bag_names[index]

        if bag_name in self.loaded_bags.keys():
            bag_dict = self.loaded_bags[bag_name]
        else:
            bag_dict = self._build_bag(bag_name)
            self.loaded_bags[bag_name] = bag_dict
        
        return_bag_dict = { key: bag_dict[key] for key in self.bag_keys if key in bag_dict }

        return TensorDict(return_bag_dict)

    def get_bag_labels(self) -> list:
        """
        Returns:
            List of bag labels.
        """
        return [ self._load_labels(name) for name in self.bag_names ]

    def get_bag_names(self) -> list:
        """
        Returns:
            List of bag names.
        """
        return self.bag_names

    def subset(self, indices: list) -> 'ProcessedMILDataset':
        """
        Create a subset of the dataset.

        Arguments:
            indices: List of indices to keep.

        Returns:
            subset_dataset: Subset of the dataset.
        """

        new_dataset = copy.deepcopy(self)
        new_dataset.bag_names = [self.bag_names[i] for i in indices]
        new_dataset.loaded_bags = { k : v for k, v in new_dataset.loaded_bags.items() if k in new_dataset.bag_names }

        return new_dataset
