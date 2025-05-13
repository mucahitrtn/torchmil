import os
import pytest
import numpy as np
import torch

from torchmil.utils import build_adj, normalize_adj, add_self_loops
from torchmil.datasets import ProcessedMILDataset


@pytest.fixture(name="mil_data")
def _mil_data():
    """
    Pytest fixture to set up and tear down temporary data for MIL dataset testing.
    """
    temp_dir = "temp_mil_data"
    os.makedirs(temp_dir, exist_ok=True)

    features_dir = os.path.join(temp_dir, "features")
    labels_dir = os.path.join(temp_dir, "labels")
    inst_labels_dir = os.path.join(temp_dir, "inst_labels")
    coords_dir = os.path.join(temp_dir, "coords")

    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(inst_labels_dir, exist_ok=True)
    os.makedirs(coords_dir, exist_ok=True)

    bag_names = ["bag1", "bag2", "bag3"]
    bag_data = {
        "bag1": {
            "features": np.array([[1, 2], [3, 4], [5, 6]]),
            "labels": np.array([1]),
            "inst_labels": np.array([[0], [1], [0]]),
            "coords": np.array([[0, 0], [1, 0], [0, 1]]),
        },
        "bag2": {
            "features": np.array([[7, 8], [9, 10]]),
            "labels": np.array([0]),
            "inst_labels": np.array([[1], [0]]),
            "coords": np.array([[2, 2], [3, 3]]),
        },
        "bag3": {
            "features": np.array([[11, 12]]),
            "labels": np.array([1]),
            "inst_labels": np.array([[1]]),
            "coords": np.array([[4, 4]]),
        },
    }

    for name, data in bag_data.items():
        np.save(os.path.join(features_dir, name + ".npy"), data["features"])
        np.save(os.path.join(labels_dir, name + ".npy"), data["labels"])
        np.save(os.path.join(inst_labels_dir, name + ".npy"), data["inst_labels"])
        np.save(os.path.join(coords_dir, name + ".npy"), data["coords"])

    yield temp_dir, features_dir, labels_dir, inst_labels_dir, coords_dir, bag_names, bag_data

    # Teardown
    for name in bag_names:
        os.remove(os.path.join(features_dir, name + ".npy"))
        os.remove(os.path.join(labels_dir, name + ".npy"))
        os.remove(os.path.join(inst_labels_dir, name + ".npy"))
        os.remove(os.path.join(coords_dir, name + ".npy"))

    os.rmdir(features_dir)
    os.rmdir(labels_dir)
    os.rmdir(inst_labels_dir)
    os.rmdir(coords_dir)
    os.rmdir(temp_dir)



def test_dataset_loading(mil_data):
    """
    Test case to verify the correct loading of the ProcessedMILDataset.
    """
    (
        temp_dir,
        features_dir,
        labels_dir,
        inst_labels_dir,
        coords_dir,
        bag_names,
        bag_data,
    ) = mil_data
    dataset = ProcessedMILDataset(
        features_path=features_dir,
        labels_path=labels_dir,
        inst_labels_path=inst_labels_dir,
        coords_path=coords_dir,
    )

    assert len(dataset) == len(bag_names)  # Check the total number of bags

    for i, bag_name in enumerate(bag_names):
        bag = dataset[i]
        expected_data = bag_data[bag_name]

        # Check that the data is loaded correctly.
        assert torch.equal(bag["X"], torch.from_numpy(expected_data["features"])), f"Features for {bag_name} do not match"
        assert torch.equal(bag["Y"], torch.from_numpy(expected_data["labels"])), f"Labels for {bag_name} do not match"
        assert torch.equal(
            bag["y_inst"], torch.from_numpy(expected_data["inst_labels"])
        ), f"Instance labels for {bag_name} do not match"
        assert torch.equal(bag["coords"], torch.from_numpy(expected_data["coords"])), f"Coordinates for {bag_name} do not match"
        assert "adj" in bag, f"Adjacency matrix is missing for {bag_name}"



def test_get_bag_labels(mil_data):
    """
    Test case to verify the correct retrieval of bag labels.
    """
    (
        temp_dir,
        features_dir,
        labels_dir,
        inst_labels_dir,
        coords_dir,
        bag_names,
        bag_data,
    ) = mil_data
    dataset = ProcessedMILDataset(
        features_path=features_dir,
        labels_path=labels_dir,
        inst_labels_path=inst_labels_dir,
        coords_path=coords_dir,
    )
    labels = dataset.get_bag_labels()
    expected_labels = [bag_data[name]["labels"] for name in dataset.bag_names]
    for i, bag_name in enumerate(dataset.bag_names):
        assert np.array_equal(labels[i], expected_labels[i]), f"Bag label for {bag_name} does not match"



def test_subset(mil_data):
    """
    Test case to verify the correct creation of a subset of the dataset.
    """
    (
        temp_dir,
        features_dir,
        labels_dir,
        inst_labels_dir,
        coords_dir,
        bag_names,
        bag_data,
    ) = mil_data

    dataset = ProcessedMILDataset(
        features_path=features_dir,
        labels_path=labels_dir,
        inst_labels_path=inst_labels_dir,
        coords_path=coords_dir,
    )
    subset_indices = [dataset.bag_names.index("bag1"), dataset.bag_names.index("bag3")]
    subset_dataset = dataset.subset(subset_indices)

    assert len(subset_dataset) == len(
        subset_indices
    ), "Subset size is incorrect"  # Check subset size
    assert subset_dataset.bag_names == ["bag1", "bag3"], "Subset bag names are incorrect"  # Check bag names

    for name in dataset.bag_names:
        bag = dataset[dataset.bag_names.index(name)]
        expected_data = bag_data[name]
        assert torch.equal(bag["X"], torch.from_numpy(expected_data["features"])), f"Features for bag {name} in subset do not match"
        assert torch.equal(bag["Y"], torch.from_numpy(expected_data["labels"])), f"Labels for bag {name} in subset do not match"
        assert torch.equal(
            bag["y_inst"], torch.from_numpy(expected_data["inst_labels"])
        ), f"Instance labels for bag {name} in subset do not match"
        assert torch.equal(bag["coords"], torch.from_numpy(expected_data["coords"])), f"Coordinates for bag {name} in subset do not match"
        assert "adj" in bag, f"Adjacency matrix is missing for bag {name} in subset"
