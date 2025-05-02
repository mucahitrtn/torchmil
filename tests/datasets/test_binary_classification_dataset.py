import os
import pytest
import numpy as np
from torchmil.datasets import BinaryClassificationDataset

def clean_temp_dir():
    """
    Helper function to clean up temporary directories.
    """
    temp_dir = "temp_binary_data"
    if os.path.exists(temp_dir):
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(temp_dir)

@pytest.fixture
def temp_binary_data():
    """
    Pytest fixture to set up and tear down temporary data for testing BinaryClassificationDataset.
    """
    temp_dir = "temp_binary_data"
    os.makedirs(temp_dir, exist_ok=True)

    features_dir = os.path.join(temp_dir, "features")
    labels_dir = os.path.join(temp_dir, "labels")
    inst_labels_dir = os.path.join(temp_dir, "inst_labels")  # Corrected variable name
    coords_dir = os.path.join(temp_dir, "coords")

    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(inst_labels_dir, exist_ok=True)  # Corrected variable name
    os.makedirs(coords_dir, exist_ok=True)

    bag_names = ["bag1", "bag2", "bag3", "bag4"]  # Added bag4 for more comprehensive testing
    bag_data = {
        "bag1": {
            "features": np.array([[1, 2], [3, 4], [5, 6]]),
            "labels": np.array(1),  # Changed to scalar
            "inst_labels": np.array([[0], [1], [0]]),
            "coords": np.array([[0, 0], [1, 0], [0, 1]]),
        },
        "bag2": {
            "features": np.array([[7, 8], [9, 10]]),
            "labels": np.array(0),  # Changed to scalar
            "inst_labels": np.array([[0], [0]]),
            "coords": np.array([[2, 2], [3, 3]]),
        },
        "bag3": {
            "features": np.array([[11, 12], [10, 9]]),
            "labels": np.array(1),  # Changed to scalar
            "inst_labels": None,
            "coords": np.array([[4, 4], [1, 1]]),
        },
        "bag4": {  # Added a bag with inconsistent instance labels
            "features": np.array([[1, 1], [2, 2], [3, 3]]),
            "labels": np.array(0),
            "inst_labels": np.array([[1], [1], [1]]),
            "coords": np.array([[5,5], [6,5], [7,5]])
        }
    }

    for name, data in bag_data.items():
        np.save(os.path.join(features_dir, name + ".npy"), data["features"])
        np.save(os.path.join(labels_dir, name + ".npy"), data["labels"])
        if data["inst_labels"] is not None:  # Added check for None
            np.save(os.path.join(inst_labels_dir, name + ".npy"), data["inst_labels"])  # Corrected path
        np.save(os.path.join(coords_dir, name + ".npy"), data["coords"])

    return temp_dir, features_dir, labels_dir, inst_labels_dir, coords_dir, bag_names, bag_data

def test_binary_classification_dataset(temp_binary_data):
    """
    Test cases for BinaryClassificationDataset.
    """
    (
        temp_dir,
        features_dir,
        labels_dir,
        inst_labels_dir,
        coords_dir,
        bag_names,
        bag_data,
    ) = temp_binary_data

    # Test dataset initialization
    dataset = BinaryClassificationDataset(
        features_path=features_dir,
        labels_path=labels_dir,
        inst_labels_path=inst_labels_dir,
        coords_path=coords_dir,
    )
    assert len(dataset) == len(bag_names), "Dataset size is incorrect"

    for i in range(len(dataset)):
        bag = dataset[i]
        bag_name = dataset.bag_names[i]
        expected_data = bag_data[bag_name]

        # Test data loading
        assert np.array_equal(bag["X"], expected_data["features"]), f"Features for {bag_name} are incorrect"
        assert np.array_equal(bag["Y"], expected_data["labels"]), f"Labels for {bag_name} are incorrect"
        assert "adj" in bag, f"Adjacency matrix is missing for {bag_name}"
        assert np.array_equal(bag["coords"], expected_data["coords"]), f"Coordinates for {bag_name} are incorrect"

        # Test instance label handling
        if expected_data["inst_labels"] is None:
            assert np.all(bag["y_inst"].numpy() == -1), f"Instance labels for {bag_name} should be -1"
        elif bag_name == "bag4":  # Check the bag with inconsistent labels
            assert np.all(bag["y_inst"].numpy() == -1), f"Instance labels for {bag_name} should be -1"
        else:
            assert np.array_equal(bag["y_inst"].numpy(), expected_data["inst_labels"].squeeze()), f"Instance labels for {bag_name} are incorrect"
    
    clean_temp_dir()
