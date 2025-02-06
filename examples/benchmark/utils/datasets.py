import warnings
import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from torchmil.datasets import WSIDataset, CTScanDataset

DATA_DIR = '/home/fran/data/datasets/'

DATASET_DIR = {
    'rsna' : f"{DATA_DIR}/RSNA_ICH/MIL_processed/",
    'panda' : f"{DATA_DIR}/PANDA/PANDA_original/",
    'camelyon16' : f"{DATA_DIR}/CAMELYON16/"
}

class WSIClassificationDataset(WSIDataset):
    def __init__(
        self,
        features_path: str,
        labels_path: str,
        patch_labels_path: str = None,
        coords_path: str = None,
        wsi_names: list = None,
        patch_size: int = 512,
        adj_with_dist: bool = False,
        norm_adj: bool = True,
    ) -> None:
        super().__init__(
            features_path=features_path,
            labels_path=labels_path,
            patch_labels_path=patch_labels_path,
            coords_path=coords_path,
            wsi_names=wsi_names,
            patch_size=patch_size,
            adj_with_dist=adj_with_dist,
            norm_adj=norm_adj,
        )

    def _load_bag(self, name: str) -> dict[str, np.ndarray]:
        bag_dict = super()._load_bag(name)

        if self.inst_labels_path is not None:
            if bag_dict['y_inst'] is None:
                if bag_dict['Y'][0] == 0:
                    bag_dict['y_inst'] = np.zeros(bag_dict['X'].shape[0])
                else:
                    warnings.warn(
                        f'Instance labels not found for {name}. Setting all to -1.')
                    bag_dict['y_inst'] = np.full(bag_dict['X'].shape[0], -1)

        return bag_dict

class CTScanClassificationDataset(CTScanDataset):
    def __init__(
        self,
        features_path: str,
        labels_path: str,
        slice_labels_path: str = None,
        ctscan_names: list = None,
        adj_with_dist: bool = False,
        norm_adj: bool = True,
    ) -> None:

        super().__init__(
            features_path=features_path,
            labels_path=labels_path,
            slice_labels_path=slice_labels_path,
            ctscan_names=ctscan_names,
            adj_with_dist=adj_with_dist,
            norm_adj=norm_adj
        )

    def _load_bag(self, name: str) -> dict[str, np.ndarray]:
        bag_dict = super()._load_bag(name)

        if self.inst_labels_path is not None:
            if bag_dict['y_inst'] is None:
                if bag_dict['label'][0] == 0:
                    bag_dict['y_inst'] = np.zeros(bag_dict['features'].shape[0])
                else:
                    warnings.warn(
                        f'Instance labels not found for {name}. Setting all to -1.')
                    bag_dict['y_inst'] = np.full(bag_dict['features'].shape[0], -1)

        return bag_dict


def load_dataset(config, mode='train_val'):
    name = config.dataset_name
    dataset_id = name.split('-')[0]

    if dataset_id == "rsna":
        # rsna-<features_dir_name>

        dataset_dir = DATASET_DIR['rsna']

        features_dir_name = name.split('-')[1]

        features_path = f'{dataset_dir}/features/{features_dir_name}/'
        labels_path = f'{dataset_dir}/labels/'
        slice_labels_path = f'{dataset_dir}/slice_labels/'
        splits_file = f'{dataset_dir}/splits.csv'

        df = pd.read_csv(splits_file)
        if mode=='train_val':
            ctscan_names = df[df['split']=='train']['bag_name'].values
        else:
            ctscan_names = df[df['split']=='test']['bag_name'].values

        dataset = CTScanClassificationDataset(
            features_path=features_path,
            labels_path=labels_path,
            slice_labels_path=slice_labels_path,
            ctscan_names=ctscan_names,
            adj_with_dist=True,
            norm_adj=True
        )

    elif dataset_id in ["panda", "camelyon16"]:
        # <dataset_id>-<patches_dir_name>-<features_dir_name>

        dataset_dir = DATASET_DIR[dataset_id]

        patches_dir_name = name.split('-')[1] # patches_<patch_size>_<whatever>
        patch_size = int(patches_dir_name.split('_')[1])
        features_dir_name = name.split('-')[2]

        features_path = f'{dataset_dir}/{patches_dir_name}/features/{features_dir_name}/'
        labels_path = f'{dataset_dir}/{patches_dir_name}/labels/'
        coords_path = f'{dataset_dir}/{patches_dir_name}/coords/'
        patch_labels_path = f'{dataset_dir}/{patches_dir_name}/patch_labels/'
        splits_file = f'{dataset_dir}/splits.csv'

        df = pd.read_csv(splits_file)
        if mode=='train_val':
            wsi_names = df[df['split']=='train']['bag_name'].values
        else:
            wsi_names = df[df['split']=='test']['bag_name'].values

        dataset = WSIClassificationDataset(
            features_path=features_path,
            labels_path=labels_path,
            patch_labels_path=patch_labels_path,
            coords_path=coords_path,
            wsi_names=wsi_names,
            patch_size=patch_size,
            adj_with_dist=True,
            norm_adj=True
        )

    else:
        raise ValueError(f"Dataset {name} not supported")

    if mode=='train_val':
        val_prop = config.val_prop
        seed = config.seed

        bags_labels = dataset.get_bag_labels()
        len_ds = len(dataset)
        
        idx = list(range(len_ds))
        idx_train, idx_val = train_test_split(idx, test_size=val_prop, random_state=seed, stratify=bags_labels)

        train_dataset = torch.utils.data.Subset(dataset, idx_train)
        val_dataset = torch.utils.data.Subset(dataset, idx_val)

        return train_dataset, val_dataset
    elif mode=='test':
        test_dataset = dataset

        return test_dataset
