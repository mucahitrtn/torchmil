import torch

import numpy as np

from collections import deque

from tensordict import TensorDict

from typing import Union


class ToyDataset(torch.utils.data.Dataset):

    r"""

    This class represents a synthetic dataset for Multiple Instance Learning (MIL) tasks.
    It generates synthetic bags of instances from a given dataset, where each bag is labeled based on the presence or absence of specific "positive" instances.
    This class is particularly useful for simulating MIL scenarios, where the goal is to learn from bags of instances rather than individual data points.

    **Bag generation.**
    The dataset generates bags by sampling instances from the input  `(data, labels)` pair.
    A bag is labeled as positive if it contains at least one instance from a predefined set of positive labels (`obj_labels`).
    The probability of generating a positive bag can be controlled via the argument `pos_class_prob`.
    The size of each bag can be specified using the argument `bag_size`.
    Additionally, each bag includes instance-level labels, indicating whether individual instances belong to the positive class.

    Each bag is returned as a dictionary (TensorDict) with the following keys:

    - X: The bag's feature matrix of shape `(bag_size, num_features)`.
    - Y: The bag's label (1 for positive, 0 for negative).
    - y_inst: The instance-level labels within the bag.

    **MNIST example.**
    We can create a MIL dataset from the original MNIST as follows:

    ```python

    import torch
    from torchvision import datasets, transforms

    # Load MNIST dataset
    mnist_train = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())

    # Extract features and labels
    data = mnist_train.data.numpy().reshape(-1, 28*28) / 255
    labels = mnist_train.targets.numpy()

    # Define positive labels
    obj_labels = [1, 2] # Digits 1 and 2 are considered positive

    # Create MIL dataset
    toy_dataset = ToyDataset(data, labels, num_bags=1000, obj_labels=obj_labels, bag_size=10)

    # Retrieve a bag
    bag = toy_dataset[0]
    X, Y, y_inst = bag['X'], bag['Y'], bag['y_inst']
    ```
    """

    def __init__(
            self,
            data : np.ndarray,
            labels : np.ndarray,
            num_bags : int,
            obj_labels : list[int],
            bag_size : Union[int, tuple[int, int]],
            pos_class_prob : float = 0.5,
            seed : int = 0
        ) -> None:
        """
        ToyMIL dataset class constructor.

        Arguments:
            data: Data matrix of shape `(num_instances, num_features)`.
            labels: Labels vector of shape `(num_instances,)`.
            num_bags: Number of bags to generate.
            obj_labels: List of labels to consider as positive.
            bag_size: Number of instances per bag. If a tuple `(min_size, max_size)` is provided, the bag size is sampled uniformly from this range.
            pos_class_prob: Probability of generating a positive bag.
            seed: Random seed.
        """

        super().__init__()

        self.data = data
        self.labels = labels
        self.num_bags = num_bags
        self.obj_labels = obj_labels
        self.bag_size = bag_size
        self.pos_class_prob = pos_class_prob

        np.random.seed(seed)
        self.bags_list = self._create_bags()

        # print(f"Expected number of bags: {self.num_bags}, Created bags: {len(self.bags_list)}")

    def _create_bags(self):
        pos_idx = np.where(np.isin(self.labels, self.obj_labels))[0]
        np.random.shuffle(pos_idx)
        neg_idx = np.where(~np.isin(self.labels, self.obj_labels))[0]
        np.random.shuffle(neg_idx)

        num_pos_bags = int(self.num_bags * self.pos_class_prob)
        num_neg_bags = self.num_bags - num_pos_bags

        pos_idx_queue = deque(pos_idx)
        neg_idx_queue = deque(neg_idx)

        bags_list = []

        for _ in range(num_pos_bags):
            data = []
            inst_labels = []
            if isinstance(self.bag_size, tuple):
                bag_size = np.random.randint(self.bag_size[0], self.bag_size[1])
            else:
                bag_size = self.bag_size
            if bag_size // 2 <= 1:
                num_positives = 1
            else:
                num_positives = np.random.randint(1, bag_size//2)
            num_negatives = bag_size - num_positives
            for _ in range(num_positives):
                a = pos_idx_queue.pop()
                data.append(self.data[a])
                inst_labels.append(self.labels[a])
                pos_idx_queue.appendleft(a)
            for _ in range(num_negatives):
                a = neg_idx_queue.pop()
                data.append(self.data[a])
                inst_labels.append(self.labels[a])
                neg_idx_queue.appendleft(a)

            idx_sort = np.argsort(inst_labels)
            data = np.stack(data)[idx_sort]
            inst_labels = np.array(inst_labels)[idx_sort]
            inst_labels = np.where(np.isin(inst_labels, self.obj_labels), 1, 0)
            label = np.max(inst_labels)

            bag_dict = TensorDict({
                'X': torch.from_numpy(data).float(),
                'Y': torch.as_tensor(label).long(),
                'y_inst': torch.from_numpy(inst_labels).long()
            })
            bags_list.append(bag_dict)

        for _ in range(num_neg_bags):
            data = []
            inst_labels = []
            if isinstance(self.bag_size, tuple):
                bag_size = np.random.randint(self.bag_size[0], self.bag_size[1])
            else:
                bag_size = self.bag_size
            for _ in range(bag_size):
                a = neg_idx_queue.pop()
                data.append(self.data[a])
                inst_labels.append(self.labels[a])
                neg_idx_queue.appendleft(a)

            idx_sort = np.argsort(inst_labels)
            data = np.stack(data)[idx_sort]
            inst_labels = np.array(inst_labels)[idx_sort]
            inst_labels = np.zeros_like(inst_labels)
            label = 0

            bag_dict = TensorDict({
                'X': torch.from_numpy(data).float(),
                'Y': torch.as_tensor(label).long(),
                'y_inst': torch.from_numpy(inst_labels).long()
            })
            bags_list.append(bag_dict)

        # Shuffle bags
        np.random.shuffle(bags_list)

        return bags_list

    def __len__(self) -> int:
        """
        Returns:
            Number of bags in the dataset
        """
        return len(self.bags_list)

    def __getitem__(self, index: int) -> TensorDict:
        """
        Arguments:
            index: Index of the bag to retrieve.

        Returns:
            bag_dict: Dictionary containing the following keys:

                - X: Bag features of shape `(bag_size, feat_dim)`.
                - Y: Label of the bag.
                - y_inst: Instance labels of the bag.
        """
        if index >= len(self.bags_list):
            raise IndexError(f"Index {index} out of range (max: {len(self.bags_list) - 1})")
        return self.bags_list[index]
