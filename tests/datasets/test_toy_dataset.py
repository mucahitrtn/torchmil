import torch
import numpy as np
from tensordict import TensorDict
import pytest
from torchmil.datasets.toy_dataset import ToyDataset

# Fixture for creating a simple ToyDataset instance
@pytest.fixture
def toy_dataset():
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.int32)
    labels = np.array([0, 1, 0, 1, 2], dtype=np.int32)
    num_bags = 5
    obj_labels = [1, 2]
    bag_size = 3
    pos_class_prob = 0.6
    seed = 42
    return ToyDataset(data, labels, num_bags, obj_labels, bag_size, pos_class_prob, seed)

def test_toydataset_initialization(toy_dataset):
    # Test case 1: Check if the dataset is initialized correctly.
    assert toy_dataset.num_bags == 5
    assert toy_dataset.obj_labels == [1, 2]
    assert toy_dataset.bag_size == 3
    assert toy_dataset.pos_class_prob == 0.6

def test_toydataset_len(toy_dataset):
    # Test case 2: Check if the length of the dataset is correct.
    assert len(toy_dataset) == 5

def test_toydataset_getitem(toy_dataset):
    # Test case 3: Check if __getitem__ returns a TensorDict with the correct keys and shapes.
    bag = toy_dataset[0]
    assert isinstance(bag, TensorDict)
    assert set(bag.keys()) == {'X', 'Y', 'y_inst'}
    assert bag['X'].shape[0] == 3  # bag_size
    assert bag['X'].shape[1] == 2  # num_features (from data)

def test_toydataset_getitem_indexerror(toy_dataset):
    # Test case 4: Check if __getitem__ raises an IndexError for out-of-bounds access.
    with pytest.raises(IndexError):
        toy_dataset[10]  # Assuming num_bags is less than 10

def test_toydataset_positive_bag_creation(toy_dataset):
    # Test case 5: Check if positive bags are created correctly.
    # Check that at least one instance label is 1, and bag label is also 1.
    for i in range(len(toy_dataset)): # Iterate through bags
        bag = toy_dataset[i]
        if bag['Y'] == 1:
            assert torch.any(bag['y_inst'] == 1)

def test_toydataset_negative_bag_creation(toy_dataset):
    # Test case 6: Check if negative bags are created correctly.
    # Check that all instance labels are 0, and bag label is also 0.

    for i in range(len(toy_dataset)): # Iterate through bags.
        bag = toy_dataset[i]
        if bag['Y'] == 0:
            assert torch.all(bag['y_inst'] == 0)

def test_toydataset_variable_bag_size():
    # Test case 7: Test with variable bag sizes.
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.int32)
    labels = np.array([0, 1, 0, 1, 2], dtype=np.int32)
    num_bags = 5
    obj_labels = [1, 2]
    bag_size = (2, 4)  # Bag size between 2 and 4
    pos_class_prob = 0.5
    seed = 42
    dataset = ToyDataset(data, labels, num_bags, obj_labels, bag_size, pos_class_prob, seed)

    for i in range(num_bags):
        bag = dataset[i]
        assert 2 <= bag['X'].shape[0] <= 4

def test_toydataset_large_num_bags():
    # Test case 8: Test with a large number of bags.  This is more of a smoke test.
    data = np.random.rand(1000, 10)
    labels = np.random.randint(0, 3, 1000)
    num_bags = 200
    obj_labels = [1, 2]
    bag_size = 5
    pos_class_prob = 0.5
    seed = 42
    dataset = ToyDataset(data, labels, num_bags, obj_labels, bag_size, pos_class_prob, seed)
    assert len(dataset) == num_bags

# def test_toydataset_edge_case_empty_data():
#     # Test case 9: Test with empty input data.
#     data = np.empty((0, 2))
#     labels = np.empty((0,))
#     num_bags = 5
#     obj_labels = [1, 2]
#     bag_size = 3
#     pos_class_prob = 0.5
#     seed = 42
#     dataset = ToyDataset(data, labels, num_bags, obj_labels, bag_size, pos_class_prob, seed)
#     assert len(dataset) == num_bags # Should still create the correct number of bags, even if they are empty.

def test_toydataset_deterministic_behavior():
    # Test case 10: Check that the dataset generates the same bags with the same seed.
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.int32)
    labels = np.array([0, 1, 0, 1, 2], dtype=np.int32)
    num_bags = 5
    obj_labels = [1, 2]
    bag_size = 3
    pos_class_prob = 0.5
    seed = 42

    dataset1 = ToyDataset(data, labels, num_bags, obj_labels, bag_size, pos_class_prob, seed)
    dataset2 = ToyDataset(data, labels, num_bags, obj_labels, bag_size, pos_class_prob, seed)

    for i in range(num_bags):
        bag1 = dataset1[i]
        bag2 = dataset2[i]
        assert torch.allclose(bag1['X'], bag2['X'])
        assert bag1['Y'] == bag2['Y']
        assert torch.allclose(bag1['y_inst'], bag2['y_inst'])
