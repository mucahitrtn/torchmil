import pytest
import numpy as np
import pandas as pd

from torchmil.datasets import CAMELYON16MILDataset  # Update with actual path


@pytest.fixture
def mock_camelyon16_data(tmp_path):
    root = tmp_path
    patch_size = 512
    features = 'UNI'
    dataset_path = root / f'patches_{patch_size}'
    features_path = dataset_path / f'features/features_{features}'
    labels_path = dataset_path / 'labels'
    patch_labels_path = dataset_path / 'patch_labels'
    coords_path = dataset_path / 'coords'

    for path in [features_path, labels_path, patch_labels_path, coords_path]:
        path.mkdir(parents=True, exist_ok=True)

    wsi_name = 'wsi1'
    np.save(features_path / f'{wsi_name}.npy', np.random.rand(4, 64))
    np.save(labels_path / f'{wsi_name}.npy', np.array(1))
    np.save(patch_labels_path / f'{wsi_name}.npy', np.random.randint(0, 2, size=(4,)))
    np.save(coords_path / f'{wsi_name}.npy', np.random.rand(4, 2))

    # Create a splits.csv
    splits = pd.DataFrame({'bag_name': [wsi_name], 'split': ['train']})
    splits.to_csv(root / 'splits.csv', index=False)

    return {
        "root": str(root),
        "features": features,
        "partition": "train",
        "patch_size": patch_size,
        "adj_with_dist": False,
        "norm_adj": True,
        "load_at_init": False
    }


def test_camelyon16_init(mock_camelyon16_data):
    dataset = CAMELYON16MILDataset(**mock_camelyon16_data)
    assert hasattr(dataset, "_load_bag")
    assert dataset.features_path.endswith("features/features_UNI/")


def test_camelyon16_load_bag(mock_camelyon16_data):
    dataset = CAMELYON16MILDataset(**mock_camelyon16_data)
    bag = dataset._load_bag("wsi1")

    assert "X" in bag
    assert "Y" in bag
    assert "y_inst" in bag
    assert "coords" in bag
    assert bag["X"].shape[0] == 4
