import numpy as np

from .binary_classification_dataset import BinaryClassificationDataset
from .ctscan_dataset import CTScanDataset

from ..utils.common import read_csv, keep_only_existing_files

class RSNAMILDataset(BinaryClassificationDataset, CTScanDataset):
    r"""
    RSNA Intracranial Hemorrhage Detection dataset for Multiple Instance Learning (MIL).

    The [original dataset](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection) has been processed to be used for MIL binary classification problems.
    It can be downloaded from [here](https://huggingface.co/datasets/Franblueee/RSNA_ICH_MIL).

    A slice is considered positive if it shows evidence of hemorrhage. The CT scan label is positive if at least one slice is positive.

    The following directory structure is expected:

        ```
        root
        ├── features
        │   ├── features_{features}
        │   │   ├── ctscan_name1.npy
        │   │   ├── ctscan_name2.npy
        │   │   └── ...
        ├── labels
        │   ├── ctscan_name1.npy
        │   ├── ctscan_name2.npy
        │   └── ...
        ├── slice_labels
        │   ├── ctscan_name1.npy
        │   ├── ctscan_name2.npy
        │   └── ...
        └── splits.csv
        ```
    """
    def __init__(
        self,
        root : str,
        features : str = 'resnet50',
        partition : str = 'train',
        bag_keys: list = ["X", "Y", "y_inst", "adj", "coords"],
        adj_with_dist: bool = False,
        norm_adj: bool = True,
        load_at_init: bool = True
    ) -> None:
        """
        Arguments:
            root: Path to the root directory of the dataset.
            features: Type of features to use. Must be one of ['resnet18', 'resnet50', 'vit_b_32']
            partition: Partition of the dataset. Must be one of ['train', 'test'].
            bag_keys: List of keys to use for the bags. Must be in ['X', 'Y', 'y_inst', 'coords'].
            adj_with_dist: If True, the adjacency matrix is built using the Euclidean distance between the patches features. If False, the adjacency matrix is binary.
            norm_adj: If True, normalize the adjacency matrix.
            load_at_init: If True, load the bags at initialization. If False, load the bags on demand.
        """
        features_path = f'{root}/features/features_{features}/'
        labels_path = f'{root}/labels/'
        slice_labels_path = f'{root}/slice_labels/'

        splits_file = f'{root}/splits.csv'
        dict_list = read_csv(splits_file)
        ctscan_names = [ row['bag_name'] for row in dict_list if row['split'] == partition]
        ctscan_names = list(set(ctscan_names))
        ctscan_names = keep_only_existing_files(features_path, ctscan_names)

        CTScanDataset.__init__(
            self,
            features_path=features_path,
            labels_path=labels_path,
            slice_labels_path=slice_labels_path,
            bag_keys=bag_keys,
            ctscan_names=ctscan_names,
            adj_with_dist=adj_with_dist,
            norm_adj=norm_adj,
            load_at_init=load_at_init
        )

    def _load_bag(self, name: str) -> dict[str, np.ndarray]:
        bag_dict = BinaryClassificationDataset._load_bag(self, name)
        bag_dict = CTScanDataset._add_coords(self, bag_dict)
        return bag_dict
