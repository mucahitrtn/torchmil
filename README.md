# torchmil

**torchmil** is a [PyTorch](https://pytorch.org/)-based library for deep Multiple Instance Learning (MIL).
It provides a simple, flexible, and extensible framework for working with MIL models and data.

It includes:

- A collection of popular [MIL models](https://franblueee.github.io/torchmil/api/models/).
- Different [PyTorch modules](https://franblueee.github.io/torchmil/api/nn/) frequently used in MIL models.
- Handy tools to deal with [MIL data](https://franblueee.github.io/torchmil/api/data/).
- A collection of popular [MIL datasets](https://franblueee.github.io/torchmil/api/datasets/).

## Installation

```bash
pip install torchmil
```

## Quick start

You can load a MIL dataset and train a MIL model in just a few lines of code:

```python
from torchmil.datasets import Camelyon16MIL
from torchmil.models import ABMIL
from torchmil.utils import Trainer
from torchmil.data import collate_fn
from torch.utils.data import DataLoader

# Load the Camelyon16 dataset
dataset = Camelyon16MIL(root='data', features='UNI')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Instantiate the ABMIL model and optimizer
model = ABMIL(in_shape=(2048,), criterion=torch.nn.BCEWithLogitsLoss()) # each model has its own criterion
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Instantiate the Trainer
trainer = Trainer(model, optimizer, device='cuda')

# Train the model
trainer.train(dataloader, epochs=10)

# Save the model
torch.save(model.state_dict(), 'model.pth')
```

## Next steps

You can take a look at the [examples](https://franblueee.github.io/torchmil/examples/) to see how to use **torchmil** in practice.
To see the full list of available models, datasets, and modules, check the [API reference](https://franblueee.github.io/torchmil/api/).

## Citation

If you find this library useful, please consider citing it:

```bibtex
@misc{torchmil,
  author = {Castro-Mac{\'\i}as, Francisco M and S{\'a}ez-Maldonado, Francisco Javier and Morales Alvarez, Pablo and Molina, Rafael},
  title = {torchmil: A PyTorch-based library for deep Multiple Instance Learning},
  year = {2025},
  howpublished = {\url{https://franblueee.github.io/torchmil/}}
}
```
