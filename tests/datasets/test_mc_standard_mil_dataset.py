import pytest
import torch
from torchmil.datasets import MCStandardMILDataset  # Update this import


@pytest.mark.parametrize("train_mode", [True, False])
def test_dataset_creation(train_mode):
    dataset = MCStandardMILDataset(D=3, num_bags=6, seed=123, train=train_mode)
    assert len(dataset) == 6
    for i in range(len(dataset)):
        bag = dataset[i]
        assert "X" in bag and "Y" in bag and "y_inst" in bag
        assert bag["X"].ndim == 2
        assert bag["y_inst"].ndim == 1
        assert bag["X"].shape[0] == bag["y_inst"].shape[0]
        assert bag["X"].shape[1] == 3


def test_label_type_and_value():
    dataset = MCStandardMILDataset(D=2, num_bags=4, train=True)
    for i in range(len(dataset)):
        Y = dataset[i]["Y"]
        assert isinstance(Y, torch.Tensor)
        assert Y.item() in (0, 1)


def test_index_out_of_range():
    dataset = MCStandardMILDataset(D=2, num_bags=3)
    with pytest.raises(IndexError):
        _ = dataset[999]


@pytest.mark.parametrize("train", [True, False])
def test_poisoning_present(train):
    dataset = MCStandardMILDataset(D=2, num_bags=10, train=train, seed=42)
    poisoned_found = False
    poison_value = -10  # Poisoning distribution mean

    for i in range(len(dataset)):
        X = dataset[i]["X"]
        if torch.any(X < -9.0):  # Poison values should be far negative
            poisoned_found = True
            break

    assert (
        poisoned_found
    ), f"No poisoned instance found in {'train' if train else 'test'} mode."


def test_positive_bag_has_both_concepts():
    dataset = MCStandardMILDataset(D=2, num_bags=10, train=True, seed=42)
    found = False
    for bag in dataset:
        if bag["Y"].item() == 1:  # Positive bag
            # We expect two kinds of positive concepts (means 2 and 3)
            means = [2, 3]
            close_to_means = [(bag["X"] - m).abs().mean(dim=1) < 1.0 for m in means]
            if torch.any(close_to_means[0]) and torch.any(close_to_means[1]):
                found = True
                break
    assert found, "No positive bag contains both concept types."
