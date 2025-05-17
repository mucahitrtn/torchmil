import numpy as np

from .binary_classification_dataset import BinaryClassificationDataset
from .wsi_dataset import WSIDataset

from ..utils.common import read_csv, keep_only_existing_files



class CAMELYON16MILDataset(BinaryClassificationDataset, WSIDataset):
    r"""
    CAMELYON16 dataset for Multiple Instance Learning (MIL).

    The [original CAMELYON16 dataset](https://camelyon17.grand-challenge.org/Data/) has been processed to be used for MIL binary classification problems.
    It can be downloaded from [here](https://huggingface.co/datasets/Franblueee/Camelyon16_MIL/).

    A patch is considered positive if at least 50% of its pixels are annotated as tumor. The WSI label is positive if at least one patch is positive.

    The following directory structure is expected:

    ```
    root
    ├── patches_{patch_size}
    │   ├── features
    │   │   ├── features_{features}
    │   │   │   ├── wsi1.npy
    │   │   │   ├── wsi2.npy
    │   │   │   └── ...
    │   ├── labels
    │   │   ├── wsi1.npy
    │   │   ├── wsi2.npy
    │   │   └── ...
    │   ├── patch_labels
    │   │   ├── wsi1.npy
    │   │   ├── wsi2.npy
    │   │   └── ...
    │   ├── coords
    │   │   ├── wsi1.npy
    │   │   ├── wsi2.npy
    │   │   └── ...
    └── splits.csv
    ```
    """
    def __init__(
        self,
        root : str,
        features : str = 'UNI',
        partition : str = 'train',
        bag_keys: list = ["X", "Y", "y_inst", "adj", "coords"],
        patch_size: int = 512,
        adj_with_dist: bool = False,
        norm_adj: bool = True,
        load_at_init: bool = True
    ) -> None:
        """
        Arguments:
            root: Path to the root directory of the dataset.
            features: Type of features to use. Must be one of ['UNI', 'resnet50_bt'].
            partition: Partition of the dataset. Must be one of ['train', 'test'].
            bag_keys: List of keys to use for the bags. Must be in ['X', 'Y', 'y_inst', 'coords'].
            patch_size: Size of the patches. Currently, only 512 is supported.
            adj_with_dist: If True, the adjacency matrix is built using the Euclidean distance between the patches features. If False, the adjacency matrix is binary.
            norm_adj: If True, normalize the adjacency matrix.
            load_at_init: If True, load the bags at initialization. If False, load the bags on demand.
        """
        features_path = f'{root}/patches_{patch_size}/features/features_{features}/'
        labels_path = f'{root}/patches_{patch_size}/labels/'
        patch_labels_path = f'{root}/patches_{patch_size}/patch_labels/'
        coords_path = f'{root}/patches_{patch_size}/coords/'

        splits_file = f'{root}/splits.csv'
        dict_list = read_csv(splits_file)
        wsi_names = [ row['bag_name'] for row in dict_list if row['split'] == partition]
        wsi_names = list(set(wsi_names))
        wsi_names = keep_only_existing_files(features_path, wsi_names)

        WSIDataset.__init__(
            self,
            features_path=features_path,
            labels_path=labels_path,
            patch_labels_path=patch_labels_path,
            coords_path=coords_path,
            wsi_names=wsi_names,
            bag_keys=bag_keys,
            patch_size=patch_size,
            adj_with_dist=adj_with_dist,
            norm_adj=norm_adj,
            load_at_init=load_at_init
        )

    def _load_bag(self, name: str) -> dict[str, np.ndarray]:
        return BinaryClassificationDataset._load_bag(self, name)
