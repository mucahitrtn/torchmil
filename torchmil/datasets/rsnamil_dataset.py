import numpy as np

from .binary_classification_dataset import BinaryClassificationDataset
from .ctscan_dataset import CTScanDataset

from ..utils.common import read_csv, keep_only_existing_files


class RSNAMILDataset(BinaryClassificationDataset, CTScanDataset):
    r"""
    RSNA Intracranial Hemorrhage Detection dataset for Multiple Instance Learning (MIL).
    Download it from [Hugging Face Datasets](https://huggingface.co/datasets/torchmil/RSNA_ICH_MIL).

    **About the original RSNA Dataset.**
    The original [RSNA-ICH dataset](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection) contains head CT scans. The task is to identify whether a CT scan contains acute intracranial hemorrhage and its subtypes. The dataset includes a label for each slice.

    **Dataset description.**
    We have preprocessed the CT scans by computing features for each slice using various feature extractors.

    - A **slice** is labeled as positive (`slice_label=1`) if it contains evidence of hemorrhage.
    - A **CT scan** is labeled as positive (`label=1`) if it contains at least one positive slice.

    This means a CT scan is considered positive if there is any evidence of hemorrhage.

    **Directory structure.**

    After extracting the contents of the `.tar.gz` archives, the following directory structure is expected:

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

    Each `.npy` file corresponds to a single CT scan. The `splits.csv` file defines train/test splits for standardized experimentation.
    """

    def __init__(
        self,
        root: str,
        features: str = "resnet50",
        partition: str = "train",
        bag_keys: list = ["X", "Y", "y_inst", "adj", "coords"],
        adj_with_dist: bool = False,
        norm_adj: bool = True,
        load_at_init: bool = True,
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
        features_path = f"{root}/features/features_{features}/"
        labels_path = f"{root}/labels/"
        slice_labels_path = f"{root}/slice_labels/"

        splits_file = f"{root}/splits.csv"
        dict_list = read_csv(splits_file)
        ctscan_names = [
            row["bag_name"] for row in dict_list if row["split"] == partition
        ]
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
            load_at_init=load_at_init,
        )

    def _load_bag(self, name: str) -> dict[str, np.ndarray]:
        bag_dict = BinaryClassificationDataset._load_bag(self, name)
        bag_dict = CTScanDataset._add_coords(self, bag_dict)
        return bag_dict
