import numpy as np
import warnings

from torchmil.datasets import ProcessedMILDataset

class BinaryClassificationDataset(ProcessedMILDataset):
    r"""
    Dataset for binary classification MIL problems. See [`torchmil.datasets.ProcessedMILDataset`](./processed_mil_dataset.md) for more information.

    For a given bag with bag label $Y$ and instance labels $\left\{ y_1, \ldots, y_N \right \}$, this dataset assumes that 
    
    \begin{gather}
        Y \in \left\{ 0, 1 \right\}, \quad y_n \in \left\{ 0, 1 \right\}, \quad \forall n \in \left\{ 1, \ldots, N \right\},\\
        Y = \max \left\{ y_1, \ldots, y_N \right\}.
    \end{gather}

    When the instance labels are not provided, they are set to -1.
    If the instance labels are provided, but they are not consistent with the bag label, a warning is issued and the instance labels are set to the bag label.
    """
    def __init__(
        self,
        features_path: str,
        labels_path: str,
        inst_labels_path: str = None,
        coords_path: str = None,
        bag_names: list = None,
        dist_thr: float = 1.5,
        adj_with_dist: bool = False,
        norm_adj: bool = True
    ) -> None:
        super().__init__(
            features_path=features_path,
            labels_path=labels_path,
            inst_labels_path=inst_labels_path,
            coords_path=coords_path,
            bag_names=bag_names,
            dist_thr=dist_thr,
            adj_with_dist=adj_with_dist,
            norm_adj=norm_adj,
        )

    def _load_inst_labels(self, name):
        inst_labels = super()._load_inst_labels(name)
        # make sure that inst_labels has shape (bag_size,)
        if inst_labels is not None:
            while inst_labels.ndim > 1:
                inst_labels = np.squeeze(inst_labels, axis=-1)
        return inst_labels

    def _load_labels(self, name):
        labels = super()._load_labels(name)
        # make sure that labels has shape ()
        labels = np.squeeze(labels)
        return labels

    def _load_bag(self, name: str) -> dict[str, np.ndarray]:
        bag_dict = super()._load_bag(name)
        bag_dict['Y'] = np.squeeze(bag_dict['Y'])

        if bag_dict['y_inst'] is None:
            if bag_dict['Y'] == 0:
                bag_dict['y_inst'] = np.zeros(bag_dict['X'].shape[0])
            else:
                warnings.warn(
                    f'Instance labels not found for bag {name}. Setting all to -1.')
                bag_dict['y_inst'] = np.full(bag_dict['X'].shape[0], -1)
        else:
            if bag_dict['Y'] != np.max(bag_dict['y_inst']):
                warnings.warn(
                    f'Instance labels are not consistent with bag label for bag {name}. Setting all instance labels to the bag label.'
                )
                bag_dict['y_inst'] = np.full(bag_dict['X'].shape[0], bag_dict['Y'])
        return bag_dict
