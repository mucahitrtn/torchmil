import pytest
import torch
from torchmil.datasets import MCStandardMILDataset


def test_mcstandardmilda_init():
    """
    Tests the initialization of MCStandardMILDataset.
    """
    dataset = MCStandardMILDataset(D=5, num_bags=10, pos_class_prob=0.7, seed=42)
    assert dataset.num_bags == 10
    assert dataset.pos_class_prob == 0.7
    assert dataset.train is True
    assert len(dataset) == 10  # Check __len__

    # Verify distributions are initialized
    assert len(dataset.pos_distr) == 2
    assert dataset.neg_distr is not None
    assert dataset.poisoning is not None


def test_mcstandardmilda_len():
    """
    Tests the __len__ method of MCStandardMILDataset.
    """
    dataset = MCStandardMILDataset(D=2, num_bags=50)
    assert len(dataset) == 50

    dataset_empty = MCStandardMILDataset(D=2, num_bags=0)
    assert len(dataset_empty) == 0


def test_mcstandardmilda_getitem():
    """
    Tests the __getitem__ method and bag structure for MCStandardMILDataset.
    """
    D_val = 3
    num_bags_val = 20
    dataset = MCStandardMILDataset(D=D_val, num_bags=num_bags_val, seed=1)

    # Test valid index
    bag = dataset[0]
    assert isinstance(bag, dict)
    assert "X" in bag
    assert "Y" in bag
    assert "y_inst" in bag

    assert bag["X"].shape[1] == D_val  # Feature dimensionality
    assert bag["Y"].ndim == 0  # Scalar label
    assert bag["y_inst"].ndim == 1  # 1D instance labels

    # Test out-of-bounds index
    with pytest.raises(IndexError):
        dataset[num_bags_val]
    with pytest.raises(IndexError):
        dataset[-num_bags_val - 1]  # Test negative index out of bounds


def test_mcstandardmilda_positive_bag_content():
    """
    Tests the content of a positive bag in training and test mode.
    """
    D_val = 2
    # Train mode: no poisoning instance
    dataset_train = MCStandardMILDataset(
        D=D_val, num_bags=1, pos_class_prob=1.0, train=True, seed=10
    )
    pos_bag_train = dataset_train[0]
    assert pos_bag_train["Y"].item() == 1  # Bag label is positive
    # Check if poisoning instance (-1) is NOT present in train mode for positive bags
    assert -1 not in pos_bag_train["y_inst"]
    assert torch.any(pos_bag_train["y_inst"] == 1)  # Must have positive instances
    assert torch.any(pos_bag_train["y_inst"] == 0)  # Must have negative instances

    # Test mode: poisoning instance present
    dataset_test = MCStandardMILDataset(
        D=D_val, num_bags=1, pos_class_prob=1.0, train=False, seed=10
    )
    pos_bag_test = dataset_test[0]
    assert pos_bag_test["Y"].item() == 1  # Bag label is positive
    # Check if poisoning instance (-1) IS present in test mode for positive bags
    assert -1 in pos_bag_test["y_inst"]
    assert torch.any(pos_bag_test["y_inst"] == 1)  # Must have positive instances
    assert torch.any(pos_bag_test["y_inst"] == 0)  # Must have negative instances


def test_mcstandardmilda_negative_bag_content():
    """
    Tests the content of a negative bag in training and test mode.
    """
    D_val = 2
    # Train mode: poisoning instance present
    dataset_train = MCStandardMILDataset(
        D=D_val, num_bags=1, pos_class_prob=0.0, train=True, seed=11
    )
    neg_bag_train = dataset_train[0]
    assert neg_bag_train["Y"].item() == 0  # Bag label is negative
    # Check if poisoning instance (-1) IS present in train mode for negative bags
    assert -1 in neg_bag_train["y_inst"]
    assert torch.any(
        neg_bag_train["y_inst"] == 1
    )  # Must have positive instances (single)
    assert torch.any(neg_bag_train["y_inst"] == 0)  # Must have negative instances

    # Test mode: no poisoning instance
    dataset_test = MCStandardMILDataset(
        D=D_val, num_bags=1, pos_class_prob=0.0, train=False, seed=11
    )
    neg_bag_test = dataset_test[0]
    assert neg_bag_test["Y"].item() == 0  # Bag label is negative
    # Check if poisoning instance (-1) is NOT present in test mode for negative bags
    assert -1 not in neg_bag_test["y_inst"]
    assert torch.any(
        neg_bag_test["y_inst"] == 1
    )  # Must have positive instances (single)
    assert torch.any(neg_bag_test["y_inst"] == 0)  # Must have negative instances


def test_mcstandardmilda_bag_counts():
    """
    Tests that the correct number of positive and negative bags are created.
    """
    D_val = 2
    num_bags_total = 100
    pos_prob = 0.6
    dataset = MCStandardMILDataset(
        D=D_val, num_bags=num_bags_total, pos_class_prob=pos_prob, seed=12
    )

    expected_pos_bags = int(num_bags_total * pos_prob)
    expected_neg_bags = num_bags_total - expected_pos_bags

    actual_pos_bags = sum(
        1 for i in range(num_bags_total) if dataset[i]["Y"].item() == 1
    )
    actual_neg_bags = sum(
        1 for i in range(num_bags_total) if dataset[i]["Y"].item() == 0
    )

    assert actual_pos_bags == expected_pos_bags
    assert actual_neg_bags == expected_neg_bags
