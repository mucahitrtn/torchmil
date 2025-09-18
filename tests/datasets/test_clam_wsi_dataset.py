import pytest

h5py = pytest.importorskip("h5py")

import numpy as np
import pandas as pd
import torch

from torchmil.datasets import CLAMWSIDataset


@pytest.fixture
def clam_wsi_setup(tmp_path):
    features_dir = tmp_path / "pt_files"
    coords_dir = tmp_path / "h5_files"
    features_dir.mkdir()
    coords_dir.mkdir()

    slide_ids = ["slide_0", "slide_1"]
    labels = ["tumor", "normal"]

    pt_features = {}
    h5_features = {}
    coords = {}

    for idx, slide in enumerate(slide_ids):
        feat = torch.randn(4 + idx, 3)
        pt_features[slide] = feat.detach().cpu().numpy()
        torch.save(feat, features_dir / f"{slide}.pt")

        coords_arr = np.random.randint(0, 64, size=(4 + idx, 2))
        coords[slide] = coords_arr
        h5_feat = np.full((4 + idx, 3), fill_value=idx, dtype=np.float32)
        h5_features[slide] = h5_feat
        with h5py.File(coords_dir / f"{slide}.h5", "w") as handle:
            handle.create_dataset("coords", data=coords_arr)
            handle.create_dataset("features", data=h5_feat)

    csv_path = tmp_path / "slides.csv"
    pd.DataFrame({"slide_id": slide_ids, "label": labels}).to_csv(
        csv_path, index=False
    )

    return {
        "csv_path": str(csv_path),
        "features_dir": str(features_dir),
        "coords_dir": str(coords_dir),
        "slide_ids": slide_ids,
        "labels": labels,
        "pt_features": pt_features,
        "h5_features": h5_features,
        "coords": coords,
    }


def test_clam_wsi_dataset_pt_features(clam_wsi_setup):
    dataset = CLAMWSIDataset(
        csv_path=clam_wsi_setup["csv_path"],
        features_dir=clam_wsi_setup["features_dir"],
        coords_dir=clam_wsi_setup["coords_dir"],
        label_map={"normal": 0, "tumor": 1},
        bag_keys=["X", "Y", "coords"],
    )

    assert len(dataset) == len(clam_wsi_setup["slide_ids"])

    bag = dataset[0]
    slide_id = clam_wsi_setup["slide_ids"][0]
    np.testing.assert_allclose(
        bag["X"].numpy(), clam_wsi_setup["pt_features"][slide_id], rtol=1e-6
    )
    assert bag["Y"].item() == 1
    np.testing.assert_array_equal(
        bag["coords"].numpy(), clam_wsi_setup["coords"][slide_id]
    )


def test_clam_wsi_dataset_h5_features(clam_wsi_setup):
    dataset = CLAMWSIDataset(
        csv_path=clam_wsi_setup["csv_path"],
        coords_dir=clam_wsi_setup["coords_dir"],
        use_h5_features=True,
        bag_keys=["X", "Y", "coords"],
    )

    bag = dataset[1]
    slide_id = clam_wsi_setup["slide_ids"][1]
    np.testing.assert_allclose(
        bag["X"].numpy(), clam_wsi_setup["h5_features"][slide_id]
    )
    assert bag["Y"].item() in {0, 1}


def test_clam_wsi_dataset_adjacency(clam_wsi_setup):
    dataset = CLAMWSIDataset(
        csv_path=clam_wsi_setup["csv_path"],
        features_dir=clam_wsi_setup["features_dir"],
        coords_dir=clam_wsi_setup["coords_dir"],
        bag_keys=["X", "Y", "coords", "adj"],
    )

    bag = dataset[0]
    assert "adj" in bag
    assert bag["adj"].is_sparse
    assert bag["adj"].shape[0] == bag["X"].shape[0]
