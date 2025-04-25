import torch
import pytest
from torchmil.datasets import SCStandardMILDataset


def test_dataset_length():
    dataset = SCStandardMILDataset(D=5, num_bags=100, B=4)
    assert len(dataset) == 100


def test_positive_negative_balance():
    dataset = SCStandardMILDataset(D=5, num_bags=20, B=3, pos_class_prob=0.25, seed=42)
    labels = torch.stack([bag["Y"] for bag in dataset])
    assert labels.sum() == 5  # 25% of 20 is 5 positives


def test_positive_bag_structure():
    dataset = SCStandardMILDataset(D=3, num_bags=5, B=2, pos_class_prob=1.0, train=True)
    bag = dataset[0]
    assert bag["Y"].item() == 1
    assert bag["X"].shape[1] == 3
    assert torch.any(bag["y_inst"] == 1)
    assert torch.any(bag["y_inst"] == 0)


def test_negative_bag_structure():
    dataset = SCStandardMILDataset(D=4, num_bags=5, B=3, pos_class_prob=0.0, train=True)
    bag = dataset[0]
    assert bag["Y"].item() == 0
    assert bag["X"].shape[1] == 4
    assert torch.all((bag["y_inst"] == 0) | (bag["y_inst"] == -1))


def test_poisoning_in_test_mode():
    dataset = SCStandardMILDataset(D=2, num_bags=5, B=2, pos_class_prob=1.0, train=False)
    bag = dataset[0]
    assert -1 in bag["y_inst"]  # Poison instance should be labeled -1


def test_poisoning_in_train_mode_negative_bag():
    dataset = SCStandardMILDataset(D=2, num_bags=5, B=2, pos_class_prob=0.0, train=True)
    bag = dataset[0]
    assert -1 in bag["y_inst"]  # Poison instance should be included


def test_index_out_of_range():
    dataset = SCStandardMILDataset(D=2, num_bags=3, B=1)
    with pytest.raises(IndexError):
        _ = dataset[3]  # Index out of bounds
