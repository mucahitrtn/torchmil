import pytest
import torch
from torchmil.data.collate import pad_tensors, collate_fn
from tensordict import TensorDict

def test_pad_tensors_single_tensor():
    # Test the pad_tensors function with a single tensor.
    # Ensure that the tensor is padded correctly and the mask is generated as expected.
    tensor = torch.tensor([[1, 2], [3, 4]])
    padded_tensor, mask = pad_tensors([tensor], padding_value=0)

    assert padded_tensor.shape == (1, 2, 2)
    assert torch.equal(padded_tensor[0], tensor)
    assert torch.equal(mask, torch.tensor([[1, 1]], dtype=torch.uint8))

def test_pad_tensors_multiple_tensors():
    # Test the pad_tensors function with multiple tensors of varying sizes.
    # Verify that all tensors are padded to the same size and the mask is correct.
    tensor1 = torch.tensor([[1, 2], [3, 4]])
    tensor2 = torch.tensor([[5, 6]])
    padded_tensor, mask = pad_tensors([tensor1, tensor2], padding_value=0)

    assert padded_tensor.shape == (2, 2, 2)
    assert torch.equal(padded_tensor[0], tensor1)
    assert torch.equal(padded_tensor[1], torch.tensor([[5, 6], [0, 0]]))
    assert torch.equal(mask, torch.tensor([[1, 1], [1, 0]], dtype=torch.uint8))

def test_collate_fn_dense_tensors():
    # Test the collate_fn function with dense tensors.
    # Check that the tensors are collated correctly into a TensorDict with proper padding and masks.
    batch_list = [
        {"data": torch.tensor([[1, 2], [3, 4]])},
        {"data": torch.tensor([[5, 6]])}
    ]
    result = collate_fn(batch_list, sparse=False)

    assert isinstance(result, TensorDict)
    assert "data" in result
    assert "mask" in result
    assert result["data"].shape == (2, 2, 2)
    assert torch.equal(result["data"][0], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(result["data"][1], torch.tensor([[5, 6], [0, 0]]))
    assert torch.equal(result["mask"], torch.tensor([[1, 1], [1, 0]], dtype=torch.uint8))

def test_collate_fn_sparse_tensors():
    # Test the collate_fn function with sparse tensors.
    # Ensure that sparse tensors are collated correctly into a TensorDict.
    sparse_tensor1 = torch.sparse_coo_tensor(
        indices=[[0, 1], [0, 1]],
        values=[1, 2],
        size=(2, 2)
    )
    sparse_tensor2 = torch.sparse_coo_tensor(
        indices=[[0], [0]],
        values=[3],
        size=(2, 2)
    )
    batch_list = [
        {"data": sparse_tensor1},
        {"data": sparse_tensor2}
    ]
    result = collate_fn(batch_list, sparse=True)

    assert isinstance(result, TensorDict)
    assert "data" in result
    assert result["data"].is_sparse
    assert result["data"].shape == (2, 2, 2)
