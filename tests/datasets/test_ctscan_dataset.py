import pytest
import numpy as np
from torchmil.datasets import CTScanDataset  # Replace with actual path


@pytest.fixture
def mock_ctscan_data(tmp_path):
    features_path = tmp_path / "features"
    labels_path = tmp_path / "labels"
    slice_labels_path = tmp_path / "slice_labels"

    for p in [features_path, labels_path, slice_labels_path]:
        p.mkdir(parents=True, exist_ok=True)

    scan_name = "scan1"
    np.save(features_path / f"{scan_name}.npy", np.random.rand(5, 64))   # 5 slices, 64 features each
    np.save(labels_path / f"{scan_name}.npy", np.array(0))
    np.save(slice_labels_path / f"{scan_name}.npy", np.random.randint(0, 2, size=(5,)))

    return {
        "features_path": str(features_path),
        "labels_path": str(labels_path),
        "slice_labels_path": str(slice_labels_path),
        "ctscan_names": [scan_name],
        "adj_with_dist": False,
        "norm_adj": True,
        "load_at_init": False
    }


def test_ctscan_dataset_init(mock_ctscan_data):
    dataset = CTScanDataset(**mock_ctscan_data)
    assert dataset.dist_thr == 1.10
    assert hasattr(dataset, "_load_bag")


def test_load_bag_coords(mock_ctscan_data):
    dataset = CTScanDataset(**mock_ctscan_data)
    bag = dataset._load_bag(mock_ctscan_data["ctscan_names"][0])

    assert "coords" in bag
    assert isinstance(bag["coords"], np.ndarray)
    assert bag["coords"].shape[0] == bag["X"].shape[0]
    assert np.array_equal(bag["coords"].flatten(), np.arange(bag["X"].shape[0]))
