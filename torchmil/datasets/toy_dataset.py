import torch

import numpy as np

from collections import deque

from tensordict import TensorDict


class ToyDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            data : np.ndarray,
            labels : np.ndarray,
            num_bags : int,
            obj_labels : list[int],
            bag_size : int,
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
            num_positives = np.random.randint(1, self.bag_size//2)
            num_negatives = self.bag_size - num_positives
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

            bag_dict = {
                'data': torch.from_numpy(data),
                'label': torch.as_tensor(label),
                'inst_labels': torch.from_numpy(inst_labels)
            }
            bags_list.append(bag_dict)

        for _ in range(num_neg_bags):
            data = []
            inst_labels = []
            for _ in range(self.bag_size):
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
                'data': torch.from_numpy(data),
                'label': torch.as_tensor(label),
                'inst_labels': torch.from_numpy(inst_labels)
            })
            bags_list.append(bag_dict)
        
        
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

                - data: Data of the bag.
                - label: Label of the bag.
                - inst_labels: Instance labels of the bag.
        """
        return self.bags_list[index]
