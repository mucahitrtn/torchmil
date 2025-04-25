import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from torchmil.datasets import PANDAMILDataset

@pytest.fixture
def mock_panda_dataset():
    with tempfile.TemporaryDirectory() as root:
        patch_size = 512
        feature_type = 'UNI'

        # Create directory structure
        features_dir = os.path.join(root, f'patches_{patch_size}/features/features_{feature_type}')
        labels_dir = os.path.join(root, f'patches_{patch_size}/labels')
        patch_labels_dir = os.path.join(root, f'patches_{patch_size}/patch_labels')
        coords_dir = os.path.join(root, f'patches_{patch_size}/coords')
        os.makedirs(features_dir)
        os.makedirs(labels_dir)
        os.makedirs(patch_labels_dir)
        os.makedirs(coords_dir)

        # Create mock files
        for wsi in ['wsi1', 'wsi2']:
            np.save(os.path.join(features_dir, f'{wsi}.npy'), np.random.rand(10, 128))
            np.save(os.path.join(labels_dir, f'{wsi}.npy'), np.array([1]))
            np.save(os.path.join(patch_labels_dir, f'{wsi}.npy'), np.random.randint(0, 2, 10))
            np.save(os.path.join(coords_dir, f'{wsi}.npy'), np.random.rand(10, 2))

        # Create splits.csv
        split_df = pd.DataFrame({
            'bag_name': ['wsi1', 'wsi2'],
            'split': ['train', 'test']
        })
        split_df.to_csv(os.path.join(root, 'splits.csv'), index=False)

        yield root

def test_dataset_initialization_train(mock_panda_dataset):
    dataset = PANDAMILDataset(root=mock_panda_dataset, partition='train', load_at_init=True)
    assert isinstance(dataset.bag_names, list)
    assert len(dataset.bag_names) == 1
    assert 'wsi1' in dataset.bag_names

def test_dataset_initialization_test(mock_panda_dataset):
    dataset = PANDAMILDataset(root=mock_panda_dataset, partition='test', load_at_init=True)
    assert 'wsi2' in dataset.bag_names

def test_load_bag_functionality(mock_panda_dataset):
    dataset = PANDAMILDataset(root=mock_panda_dataset, partition='train', load_at_init=False)
    bag = dataset._load_bag('wsi1')
    assert isinstance(bag, dict)
    assert 'X' in bag
    assert isinstance(bag['X'], np.ndarray)
    assert bag['X'].shape[1] == 128

def test_missing_files_are_filtered(mock_panda_dataset):
    # Create a dummy split with non-existent files
    df = pd.DataFrame({
        'bag_name': ['wsi1', 'wsi_nonexistent'],
        'split': ['train', 'train']
    })
    df.to_csv(os.path.join(mock_panda_dataset, 'splits.csv'), index=False)

    dataset = PANDAMILDataset
