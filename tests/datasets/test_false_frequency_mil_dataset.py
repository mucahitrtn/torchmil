import pytest
import torch
from torchmil.datasets import FalseFrequencyMILDataset  # Update with actual module path


@pytest.fixture
def mock_dataset():
    return FalseFrequencyMILDataset(D=4, num_bags=5, train=True, seed=42)


def test_dataset_length(mock_dataset):
    assert len(mock_dataset) == 5


def test_getitem_returns_tensordict(mock_dataset):
    bag = mock_dataset[0]
    assert isinstance(bag, dict) or hasattr(bag, "get")  # Works for TensorDict too
    assert "X" in bag and "Y" in bag and "y_inst" in bag


def test_instance_shapes(mock_dataset):
    bag = mock_dataset[0]
    X = bag["X"]
    y_inst = bag["y_inst"]
    assert X.ndim == 2
    assert y_inst.ndim == 1
    assert X.shape[0] == y_inst.shape[0]


def test_feature_dimension(mock_dataset):
    bag = mock_dataset[0]
    assert bag["X"].shape[1] == 4  # feature dimension


def test_label_type(mock_dataset):
    bag = mock_dataset[0]
    assert isinstance(bag["Y"], torch.Tensor)
    assert bag["Y"].dtype in (torch.int64, torch.float32)


def test_out_of_bounds_index(mock_dataset):
    with pytest.raises(IndexError):
        _ = mock_dataset[999]
