# torchmil

**torchmil** is a [PyTorch](https://pytorch.org/)-based library for deep Multiple Instance Learning (MIL). 
It provides a simple, flexible, and extensible framework for working with MIL models and data. 

It includes:

- A collection of popular [MIL models](api/models/index.md). 
- Different [PyTorch modules](api/nn/index.md) frequently used in MIL models.
- Handy tools to deal with [MIL data](api/data/index.md).
- A collection of popular [MIL datasets](api/datasets/index.md).

## Installation

```bash
pip install torchmil
```

## Quick start

You can load a MIL dataset and train a MIL model in just a few lines of code:

```python
from torchmil.datasets import Camelyon16
from torchmil.models import ABMIL
from torchmil.utils import Trainer
from torch.utils.data import DataLoader

# Load the Camelyon16 dataset
dataset = Camelyon16(root='data', features='UNI')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Instantiate the ABMIL model, optimizer, and criterion
model = ABMIL(in_shape=(2048,), out_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

# Instantiate the Trainer
trainer = Trainer(model, optimizer, criterion, device='cuda')

# Train the model
trainer.train(dataloader, epochs=10)

# Save the model
torch.save(model.state_dict(), 'model.pth')
```

## Next steps

Head to the [get started](getting_started.md) guide to learn more about **torchmil**.
Also, yoy can take a look at the [examples](examples/index.md) to see how to use **torchmil** in practice.
To see the full list of available models, datasets, and modules, check the [API reference](api/index.md).

## Citation

If you find this library useful, please consider citing it:

```bibtex
@misc{torchmil,
  author = {},
  title = {torchmil: A PyTorch-based library for deep Multiple Instance Learning},
  year = {},
  publisher = {},
  journal = {}
}
```

