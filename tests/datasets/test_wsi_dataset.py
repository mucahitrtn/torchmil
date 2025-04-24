import pytest
import numpy as np
from torchmil.datasets import WSIDataset  # Update to actual import path


@pytest.fixture
def mock_wsi_data(tmp_path):
    features_path = tmp_path / "features"
    labels_path = tmp_path / "labels"
    coords_path = tmp_path / "coords"
    inst_labels_path = tmp_path / "patch_labels"

    for p in [features_path, labels_path, coords_path, inst_labels_path]:
        p.mkdir(parents=True, exist_ok=True)

    wsi_name = "sample"
    np.save(features_path / f"{wsi_name}.npy", np.random.rand(10, 128))
    np.save(labels_path / f"{wsi_name}.npy", np.array(1))
    np.save(coords_path / f"{wsi_name}.npy", np.random.randint(0, 1000, size=(10, 2)))
    np.save(inst_labels_path / f"{wsi_name}.npy", np.random.randint(0, 2, size=(10,)))

    return {
        "features_path": str(features_path),
        "labels_path": str(labels_path),
        "coords_path": str(coords_path),
        "patch_labels_path": str(inst_labels_path),
        "wsi_names": [wsi_name],
        "patch_size": 512,
        "adj_with_dist": False,
        "norm_adj": True,
        "load_at_init": False,
    }


def test_wsi_dataset_init(mock_wsi_data):
    dataset = WSIDataset(**mock_wsi_data)
    assert dataset.patch_size == 512
    assert hasattr(dataset, "_load_coords")


def test_load_coords_adjustment(mock_wsi_data):
    dataset = WSIDataset(**mock_wsi_data)
    coords = dataset._load_coords(mock_wsi_data["wsi_names"][0])
    assert coords is not None
    assert isinstance(coords, np.ndarray)
    assert coords.min() == 0  # coordinates should be normalized
    assert coords.dtype == np.int_
