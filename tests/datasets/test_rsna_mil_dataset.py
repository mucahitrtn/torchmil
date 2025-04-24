import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from torchmil.datasets import RSNAMILDataset  # Update this import with the actual module path


@pytest.fixture
def mock_rsna_dataset(tmp_path):
    root = tmp_path
    features = 'resnet50'
    features_path = root / f'features/features_{features}'
    labels_path = root / 'labels'
    slice_labels_path = root / 'slice_labels'

    for path in [features_path, labels_path, slice_labels_path]:
        path.mkdir(parents=True, exist_ok=True)

    ctscan_name = 'ct1'
    np.save(features_path / f'{ctscan_name}.npy', np.random.rand(5, 128))  # 5 slices with 128 features
    np.save(labels_path / f'{ctscan_name}.npy', np.array(1))
    np.save(slice_labels_path / f'{ctscan_name}.npy', np.random.randint(0, 2, size=(5,)))

    # Create a splits.csv
    splits = pd.DataFrame({'bag_name': [ctscan_name], 'split': ['train']})
    splits.to_csv(root / 'splits.csv', index=False)

    return {
        "root": str(root),
        "features": features,
        "partition": "train",
        "adj_with_dist": False,
        "norm_adj": True,
        "load_at_init": False
    }


def test_rsna_init(mock_rsna_dataset):
    dataset = RSNAMILDataset(**mock_rsna_dataset)
    assert hasattr(dataset, '_load_bag')
    assert dataset.features_path.endswith("features/features_resnet50/")


def test_rsna_load_bag(mock_rsna_dataset):
    dataset = RSNAMILDataset(**mock_rsna_dataset)
    bag = dataset._load_bag("ct1")

    assert isinstance(bag, dict)
    assert "X" in bag
    assert "Y" in bag
    assert "y_inst" in bag
    assert "coords" in bag
    assert bag["X"].shape[0] == 5
    assert bag["coords"].shape[0] == 5
