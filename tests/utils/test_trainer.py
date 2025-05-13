import torch
import pytest
from torch.utils.data import DataLoader, Dataset
from torchmil.models import ABMIL
from torchmil.utils import Trainer  
from tensordict import TensorDict


# Dummy dataset for MIL
class DummyMILData(Dataset):
    def __init__(self, n_bags=10, bag_size=5, n_features=16):
        self.n_bags = n_bags
        self.bag_size = bag_size
        self.n_features = n_features

    def __len__(self):
        return self.n_bags

    def __getitem__(self, idx):
        bag = torch.randn(self.bag_size, self.n_features)
        label = torch.randint(0, 2, (1,), dtype=torch.float32)
        return {'X': bag, 'Y': label.squeeze()}

# Custom collate_fn for MIL batching
def collate_fn(batch):
    return TensorDict({
        'X': [item['X'] for item in batch],
        'Y': torch.stack([item['Y'] for item in batch])
    })

@pytest.fixture
def dummy_dataloader():
    dataset = DummyMILData()
    return DataLoader(dataset, batch_size=2, collate_fn=collate_fn)


def test_trainer_runs(dummy_dataloader):
    model = ABMIL(in_shape=(16,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
        verbose=False,
        disable_pbar=True,
        early_stop_patience=2
    )

    # Just 1 epoch for a sanity check
    trainer.train(
        max_epochs=1,
        train_dataloader=dummy_dataloader,
        val_dataloader=dummy_dataloader,
        test_dataloader=dummy_dataloader
    )

    best_model_dict = trainer.get_best_model_state_dict()
    assert isinstance(best_model_dict, dict)

    best_model = trainer.get_best_model()
    assert isinstance(best_model, ABMIL)