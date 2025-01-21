import torch
import torchvision

import numpy as np

from scipy.sparse import csgraph

class MNISTMILDataset(torch.utils.data.Dataset):
    def __init__(self, subset="train", bag_size=9, obj_label=1, **kwargs) -> None:
        super().__init__()
        self.subset = subset
        self.bag_size = bag_size
        self.obj_label = obj_label
        dataset = torchvision.datasets.MNIST(root='/data/data_fran/mnist/', train=(subset=="train"), download=True, transform=torchvision.transforms.ToTensor())
        dataset.targets = dataset.targets.numpy().astype(np.int32)
        dataset.data = dataset.data.numpy().astype(np.float32)
        self.create_bags(dataset, bag_size, obj_label)
    
    def create_bags(self, dataset, bag_size, obj_label=1):
        self.data = []
        self.bag_labels = []
        self.inst_labels = []
        self.L_mat = []
        num_bags = len(dataset.targets) // bag_size

        idx_vec = np.arange(len(dataset.targets))
        np.random.shuffle(np.arange(len(dataset.targets)))
        dataset.data = dataset.data[idx_vec]
        dataset.targets = dataset.targets[idx_vec]

        for i in range(num_bags):
            low = i * bag_size
            high = (i+1) * bag_size
            if high > len(dataset.targets):
                high = len(dataset.targets)
            self.data.append(np.expand_dims(dataset.data[low:high]/255.0, axis=1))
            y_labels = (dataset.targets[low:high]==obj_label)*1
            self.bag_labels.append(np.max(y_labels))
            self.inst_labels.append(y_labels)
            self.L_mat.append(csgraph.laplacian(np.eye(high-low), normed=True))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float(), torch.as_tensor(self.bag_labels[index]).float(), torch.from_numpy(self.inst_labels[index]).float(), torch.from_numpy(self.L_mat[index]).float()
