import sys
sys.path.append('/work/work_fran/torchmil')

import torch
from torchmil.data.collate import collate_fn
from tensordict import TensorDict

def test_collate_fn():
    batch_list = [
        {
            'features': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            'label': torch.tensor(0),
            'inst_labels': torch.tensor([0, 1]),
            'edge_index': torch.tensor([[0, 1], [1, 0]]),
            'edge_weight': torch.tensor([1.0, 1.0]),
            'coords': torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        },
        {
            'features': torch.tensor([[5.0, 6.0]]),
            'label': torch.tensor(1),
            'inst_labels': torch.tensor([1]),
            'edge_index': torch.tensor([[0, 0]]).t(),
            'edge_weight': torch.tensor([1.0]),
            'coords': torch.tensor([[2.0, 2.0]])
        }
    ]

    result = collate_fn(batch_list, sparse=True)

    assert isinstance(result, TensorDict)
    assert 'features' in result
    assert 'label' in result
    assert 'inst_labels' in result
    assert 'adj' in result
    assert 'coords' in result
    assert 'mask' in result

    assert result['features'].shape == (2, 2, 2)
    assert result['label'].shape == (2,)
    assert result['inst_labels'].shape == (2, 2)
    assert result['adj'].shape == (2, 2, 2)
    assert result['coords'].shape == (2, 2, 2)
    assert result['mask'].shape == (2, 2)

    assert torch.equal(result['features'][0], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(result['features'][1], torch.tensor([[5.0, 6.0], [0.0, 0.0]]))
    assert torch.equal(result['label'], torch.tensor([0, 1]))
    assert torch.equal(result['inst_labels'][0], torch.tensor([0, 1]))
    assert torch.equal(result['inst_labels'][1], torch.tensor([1, 0]))
    assert torch.equal(result['coords'][0], torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    assert torch.equal(result['coords'][1], torch.tensor([[2.0, 2.0], [0.0, 0.0]]))
    assert torch.equal(result['mask'][0], torch.tensor([1, 1]))
    assert torch.equal(result['mask'][1], torch.tensor([1, 0]))

    assert result['adj'][0].to_dense().shape == (2, 2)
    assert result['adj'][1].to_dense().shape == (2, 2)

if __name__ == "__main__":
    test_collate_fn()
    print("All tests passed.")