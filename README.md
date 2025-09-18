# torchmil



<p align="center">
    <a align="center">
        <img align="middle" src="./imgs/torchmil_logo.png" width="700" />
    </a>
</p>

<p align="middle">
    <img src="https://img.shields.io/badge/license-Apache_2.0-blue" alt="License">
    <a href="https://codecov.io/gh/franblueee/torchmil">
    <img src="https://codecov.io/gh/franblueee/torchmil/branch/main/graph/badge.svg"/>
    </a>
    <a href="#"><img src="https://badgen.net/github/contributors/Franblueee/torchmil" alt="Contributors"></a>
    <a href="#"><img src="https://badgen.net/github/release/Franblueee/torchmil" alt="Release"></a>

</p>

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

## Loading CLAM-style WSI features

If you already processed Whole Slide Images with
[CLAM](https://github.com/mahmoodlab/CLAM) and stored the bags as `.pt` feature
files plus optional `.h5` coordinate files, you can load them with
`CLAMWSIDataset`:

```python
from torchmil.datasets import CLAMWSIDataset

dataset = CLAMWSIDataset(
    csv_path="dataset/slides.csv",          # slide metadata exported by CLAM
    features_dir="dataset/pt_files",        # optional: one {slide_id}.pt per slide
    coords_dir="dataset/h5_files",          # optional: one {slide_id}.h5 per slide
    label_map={"normal": 0, "tumor": 1},   # map slide labels to integers
    bag_keys=["X", "Y", "coords", "adj"],
)

bag = dataset[0]
print(bag.keys())  # -> ['X', 'Y', 'coords', 'adj']
```

Set `use_h5_features=True` when the CLAM export only contains `.h5` files (the
dataset will read both the features and coordinates from them). The adjacency
matrix is built automatically whenever coordinates are available.

## Next steps

You can take a look at the [examples](https://franblueee.github.io/torchmil/examples/) to see how to use **torchmil** in practice.
To see the full list of available models, datasets, and modules, check the [API reference](https://franblueee.github.io/torchmil/api/).

## Contributing to torchmil

We welcome contributions to **torchmil**! There several ways you can contribute:

- Reporting bugs or issues you encounter while using the library, asking questions, or requesting new features: use the [Github issues](https://github.com/Franblueee/torchmil/issues).
- Improving the documentation: if you find any part of the documentation unclear or incomplete, feel free to submit a pull request with improvements.
- If you have a new model, dataset, or utility that you think would be useful for the community, please consider submitting a pull request to add it to the library.

Take a look at [CONTRIBUTING.md](https://github.com/Franblueee/torchmil/blob/main/CONTRIBUTING.md) for more details on how to contribute.

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
