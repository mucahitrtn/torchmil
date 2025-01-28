import numpy as np

import warnings

from .processed_mil_dataset import ProcessedMILDataset

class WSIDataset(ProcessedMILDataset):
    def __init__(
        self,
        features_path: str,
        labels_path: str,
        inst_labels_path: str = None,
        coords_path: str = None,
        patch_size: int = 512,
        norm_adj: bool = True,
    ) -> None:
        """
        Class constructor.

        Arguments:
            features_path: Path to the directory containing the features.
            labels_path: Path to the directory containing the bag labels.
            inst_labels_path: Path to the directory containing the instance labels.
            coords_path: Path to the directory containing the coordinates.
            patch_size: Size of the patches.
            norm_adj: If True, normalize the adjacency matrix.
        """
        self.patch_size = patch_size
        dist_thr = np.sqrt(2.0) * patch_size
        super().__init__(
            features_path,
            labels_path,
            inst_labels_path,
            coords_path,
            dist_thr=dist_thr,
            norm_adj=norm_adj,
        )

    def _load_bag(self, name: str) -> dict[str, np.ndarray]:
        """
        Load a bag from disk.

        Arguments:
            name: Name of the bag to load.

        Returns:
            bag_dict: Dictionary containing the features, label, instance labels and coordinates of the bag.
        """
        bag_dict = super()._load_bag(name)

        if self.inst_labels_path is not None:
            if bag_dict['inst_labels'] is None:
                if bag_dict['label'][0] == 0:
                    bag_dict['inst_labels'] = np.zeros(bag_dict['features'].shape[0])
                else:
                    warnings.warn(
                        f'Instance labels not found for {name}. Setting all to -1.')
                    bag_dict['inst_labels'] = np.full(bag_dict['features'].shape[0], -1)

        return bag_dict