from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch

from .processed_mil_dataset import ProcessedMILDataset

try:  # pragma: no cover - simple import guard
    import h5py
except ModuleNotFoundError as import_error:  # pragma: no cover - handled lazily
    h5py = None
    _H5PY_IMPORT_ERROR = import_error
else:  # pragma: no cover - executed when dependency available
    _H5PY_IMPORT_ERROR = None


class CLAMWSIDataset(ProcessedMILDataset):
    r"""Dataset for CLAM-style processed WSIs.

    This dataset loads MIL bags that have been pre-processed with the
    [CLAM](https://github.com/mahmoodlab/CLAM) pipeline. CLAM stores the bag
    features as ``.pt`` files and the spatial coordinates (and optionally the
    features) as ``.h5`` files, while the metadata (e.g., the slide level
    labels) is tracked inside a CSV file. This class bridges that file layout
    with :class:`~torchmil.datasets.ProcessedMILDataset` so that the bags can be
    readily consumed by TorchMIL models.

    Parameters
    ----------
    csv_path:
        Path to the CSV file containing the slide metadata. The CSV must
        contain at least two columns with the slide identifiers and the
        corresponding labels.
    features_dir:
        Directory that stores the ``.pt`` feature files produced by CLAM.
        File names are expected to follow ``{slide_id}.pt``. Provide ``None``
        when the bag features should be read from the ``.h5`` files
        (``use_h5_features=True``).
    coords_dir:
        Directory that stores the ``.h5`` files with the instance coordinates
        (and optionally the features). File names are expected to follow
        ``{slide_id}.h5``. It must be provided when ``"coords"`` or ``"adj"``
        are requested in ``bag_keys`` or when ``use_h5_features=True``.
    slide_id_col:
        Name of the column in ``csv_path`` that identifies the slides.
    label_col:
        Name of the column in ``csv_path`` that stores the slide labels.
    wsi_names:
        Optional list with the subset of slide identifiers to load. When
        provided, only those slides are kept.
    label_map:
        Optional mapping from label values in ``label_col`` to integers. When
        omitted the labels are sorted and mapped to ``range(n_classes)``.
    bag_keys:
        Keys that should be returned for each bag. They follow the same
        semantics as :class:`~torchmil.datasets.ProcessedMILDataset`. By
        default the dataset returns the features (``"X"``), the slide label
        (``"Y"``) and the coordinates (``"coords"``). When ``coords_dir`` is
        ``None`` the keys related to coordinates (``"coords"`` and ``"adj"``)
        are automatically discarded.
    use_h5_features:
        If ``True``, the features are loaded from the ``.h5`` files instead of
        the ``.pt`` files. In that case ``coords_dir`` must be provided because
        the features are expected to live in the same file as the coordinates.
    pt_feature_key:
        Optional key that identifies the tensor containing the features when a
        ``.pt`` file stores a dictionary. When ``None`` a few common keys are
        tried (``"features"``, ``"feats"`` and ``"data"``).
    pt_feature_index:
        Index that should be used when the ``.pt`` file contains a list or a
        tuple and the features need to be selected from it. The default selects
        the first element.
    h5_feature_key:
        Dataset name inside the ``.h5`` files that stores the instance
        features. Only used when ``use_h5_features=True``.
    h5_coords_key:
        Dataset name inside the ``.h5`` files that stores the instance
        coordinates.
    dist_thr, adj_with_dist, norm_adj, load_at_init:
        See :class:`~torchmil.datasets.ProcessedMILDataset` for a description of
        these parameters.
    """

    def __init__(
        self,
        csv_path: str,
        features_dir: Optional[str] = None,
        coords_dir: Optional[str] = None,
        slide_id_col: str = "slide_id",
        label_col: str = "label",
        wsi_names: Optional[Iterable[str]] = None,
        label_map: Optional[dict] = None,
        bag_keys: Optional[list[str]] = None,
        use_h5_features: bool = False,
        pt_feature_key: Optional[str] = None,
        pt_feature_index: int = 0,
        h5_feature_key: str = "features",
        h5_coords_key: str = "coords",
        dist_thr: Optional[float] = None,
        adj_with_dist: bool = False,
        norm_adj: bool = True,
        load_at_init: bool = False,
    ) -> None:
        self.csv_path = csv_path
        self.slide_id_col = slide_id_col
        self.label_col = label_col
        self.use_h5_features = use_h5_features
        self.pt_feature_key = pt_feature_key
        self.pt_feature_index = pt_feature_index
        self.h5_feature_key = h5_feature_key
        self.h5_coords_key = h5_coords_key
        self.pt_features_path = features_dir
        self.coords_path = coords_dir

        if bag_keys is None:
            bag_keys = ["X", "Y", "coords"]
        else:
            bag_keys = list(dict.fromkeys(bag_keys))

        if coords_dir is None:
            bag_keys = [key for key in bag_keys if key not in {"coords", "adj"}]

        if (coords_dir is not None or use_h5_features) and h5py is None:
            raise ImportError(
                "h5py is required to handle '.h5' files. Please install it to use "
                "CLAMWSIDataset with coordinates or HDF5 features."
            ) from _H5PY_IMPORT_ERROR

        if "X" in bag_keys and not use_h5_features and features_dir is None:
            raise ValueError(
                "features_dir must be provided when 'X' is requested and use_h5_features is False"
            )

        if use_h5_features and coords_dir is None:
            raise ValueError(
                "coords_dir must be provided when use_h5_features=True because features are read from the h5 files"
            )

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        slide_df = pd.read_csv(csv_path)
        if slide_id_col not in slide_df.columns:
            raise ValueError(
                f"Column '{slide_id_col}' not found in the CSV file. Available columns: {slide_df.columns.tolist()}"
            )
        if label_col not in slide_df.columns:
            raise ValueError(
                f"Column '{label_col}' not found in the CSV file. Available columns: {slide_df.columns.tolist()}"
            )

        slide_df = slide_df[[slide_id_col, label_col]].dropna()
        slide_df[slide_id_col] = slide_df[slide_id_col].astype(str)

        if wsi_names is not None:
            wsi_names = [str(name) for name in wsi_names]
            missing = set(wsi_names) - set(slide_df[slide_id_col])
            if missing:
                raise ValueError(
                    f"The following slide identifiers were not found in the CSV file: {sorted(missing)}"
                )
            slide_df = slide_df[slide_df[slide_id_col].isin(wsi_names)]

        if slide_df.empty:
            raise ValueError("No slides available after applying the provided filters")

        if label_map is None:
            unique_labels = sorted(slide_df[label_col].unique())
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            missing_labels = set(slide_df[label_col].unique()) - set(label_map.keys())
            if missing_labels:
                raise ValueError(
                    "label_map is missing values for: " + ", ".join(map(str, sorted(missing_labels)))
                )

        slide_df["__label_id"] = slide_df[label_col].map(label_map)
        if slide_df["__label_id"].isnull().any():
            raise ValueError("label_map produced NaN values; please double-check it")

        bag_names = slide_df[slide_id_col].tolist()

        if not bag_names:
            raise ValueError("No slide identifiers found")

        if "X" in bag_keys and not use_h5_features:
            missing_features = [
                name
                for name in bag_names
                if not os.path.exists(os.path.join(features_dir, f"{name}.pt"))
            ]
            if missing_features:
                raise FileNotFoundError(
                    "The following feature files were not found: "
                    + ", ".join(sorted(missing_features))
                )

        if coords_dir is not None:
            missing_coords = [
                name
                for name in bag_names
                if not os.path.exists(os.path.join(coords_dir, f"{name}.h5"))
            ]
            if missing_coords:
                raise FileNotFoundError(
                    "The following coordinate files were not found: "
                    + ", ".join(sorted(missing_coords))
                )

        self.labels_dict = dict(zip(bag_names, slide_df["__label_id"].astype(int)))
        self.label_map = label_map
        self.slide_metadata = slide_df.drop(columns="__label_id").reset_index(drop=True)

        if dist_thr is None:
            dist_thr = float(np.sqrt(2.0))

        features_path = coords_dir if use_h5_features else features_dir
        super().__init__(
            features_path=features_path,
            labels_path=csv_path,
            coords_path=coords_dir,
            bag_names=bag_names,
            bag_keys=bag_keys,
            dist_thr=dist_thr,
            adj_with_dist=adj_with_dist,
            norm_adj=norm_adj,
            load_at_init=load_at_init,
        )

    def _load_features(self, name: str) -> np.ndarray:
        if self.use_h5_features:
            h5_file = os.path.join(self.coords_path, f"{name}.h5")
            with h5py.File(h5_file, "r") as handle:
                if self.h5_feature_key not in handle:
                    raise KeyError(
                        f"Dataset '{self.h5_feature_key}' not found inside {h5_file}"
                    )
                features = handle[self.h5_feature_key][:]
        else:
            feature_file = os.path.join(self.pt_features_path, f"{name}.pt")
            data = torch.load(feature_file)
            features = self._extract_pt_features(data, feature_file)

        features = np.asarray(features)
        return features

    def _extract_pt_features(self, data, feature_file: str) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, (list, tuple)):
            if len(data) <= self.pt_feature_index:
                raise IndexError(
                    f"Cannot extract index {self.pt_feature_index} from {feature_file}; "
                    f"it only contains {len(data)} elements"
                )
            selected = data[self.pt_feature_index]
            if isinstance(selected, torch.Tensor):
                return selected.detach().cpu().numpy()
            return np.asarray(selected)
        if isinstance(data, dict):
            if self.pt_feature_key is not None:
                if self.pt_feature_key not in data:
                    raise KeyError(
                        f"Key '{self.pt_feature_key}' not found in {feature_file}"
                    )
                value = data[self.pt_feature_key]
            else:
                for candidate in ("features", "feats", "data"):
                    if candidate in data:
                        value = data[candidate]
                        break
                else:
                    raise KeyError(
                        f"Could not infer the key containing the features in {feature_file}. "
                        "Please provide 'pt_feature_key' explicitly."
                    )
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)
        return np.asarray(data)

    def _load_labels(self, name: str) -> np.ndarray:
        if name not in self.labels_dict:
            raise KeyError(f"Label for slide '{name}' not found")
        return np.array(self.labels_dict[name], dtype=np.int64)

    def _load_coords(self, name: str) -> Optional[np.ndarray]:
        if self.coords_path is None:
            return None
        h5_file = os.path.join(self.coords_path, f"{name}.h5")
        with h5py.File(h5_file, "r") as handle:
            if self.h5_coords_key not in handle:
                raise KeyError(
                    f"Dataset '{self.h5_coords_key}' not found inside {h5_file}"
                )
            coords = handle[self.h5_coords_key][:]
        return coords
