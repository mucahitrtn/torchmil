# CLAM WSI dataset

The `CLAMWSIDataset` bridges pre-processed Whole Slide Image (WSI) bags generated
with the [CLAM](https://github.com/mahmoodlab/CLAM) pipeline and the
`ProcessedMILDataset` interface provided by **torchmil**. It understands the file
layout used by CLAM (CSV metadata plus `.pt` features and optional `.h5`
coordinate files) and returns bags that can be fed directly to MIL models.

## Expected file structure

```text
dataset/
├── slides.csv            # slide-level metadata with labels
├── pt_files/             # optional; one `{slide_id}.pt` file per slide
│   ├── slide_a.pt
│   └── slide_b.pt
└── h5_files/             # optional; one `{slide_id}.h5` file per slide
    ├── slide_a.h5
    └── slide_b.h5
```

- ``slides.csv`` must contain at least two columns: one with the slide
  identifiers (``slide_id`` by default) and another one with the labels
  (``label`` by default). Extra columns are preserved in
  ``CLAMWSIDataset.slide_metadata`` for downstream usage.
- ``pt_files`` should contain the CLAM features exported as PyTorch tensors. The
  dataset can also read dictionaries (``{"features": tensor}``) or sequences;
  use ``pt_feature_key``/``pt_feature_index`` when you need to select a
  particular value.
- ``h5_files`` should provide the instance coordinates in a dataset named
  ``coords`` by default. When ``use_h5_features=True`` the features are read from
  the ``features`` dataset within the same file.

Provide ``features_dir`` when you want to load the bags from the `.pt` files,
``coords_dir`` when the coordinates (or the adjacency matrix) are required, and
both directories when you need features and coordinates simultaneously.

## Quick start

```python
from torchmil.datasets import CLAMWSIDataset

# Load CLAM features stored as .pt tensors and the coordinates from .h5 files
clam_dataset = CLAMWSIDataset(
    csv_path="dataset/slides.csv",
    features_dir="dataset/pt_files",
    coords_dir="dataset/h5_files",
    label_map={"normal": 0, "tumor": 1},  # map CSV labels to integers
    bag_keys=["X", "Y", "coords", "adj"],  # request features, labels, coords, and adjacency
)

bag = clam_dataset[0]
print(bag.keys())  # -> ['X', 'Y', 'coords', 'adj']
print(bag["X"].shape)  # instance features
print(bag["Y"])        # slide-level label
```

Use ``wsi_names`` to focus on a subset of slides, and provide a custom
``label_map`` when your CSV labels are strings. Setting ``bag_keys`` controls
which tensors are returned for each bag. When the coordinates are available, the
adjacency matrix is built automatically using the Euclidean distances between
patches (see the parameters ``dist_thr``, ``adj_with_dist`` and ``norm_adj`` for
additional control).

### Loading features from `.h5`

If your CLAM export only contains `.h5` files, enable ``use_h5_features`` and
omit ``features_dir``:

```python
clam_dataset = CLAMWSIDataset(
    csv_path="dataset/slides.csv",
    coords_dir="dataset/h5_files",
    use_h5_features=True,
    bag_keys=["X", "Y"],
)
```

The dataset will read both the features and the coordinates from each HDF5 file
(using the datasets named ``features`` and ``coords`` by default).

## API reference

::: torchmil.datasets.CLAMWSIDataset
    options:
        members:
            - __init__
