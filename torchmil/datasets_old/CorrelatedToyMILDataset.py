import torch
import torchvision

import numpy as np

from collections import deque
from scipy.sparse import csgraph

class CorrelatedToyMILDataset(torch.utils.data.Dataset):
    def __init__(self, name, subset, num_bags, obj_labels, bag_size, use_inst_labels=False, seed=0) -> None:
        super().__init__()
        self.name = name
        if self.name == 'cifar100':
            self.dataset = torchvision.datasets.CIFAR100(root='/home/fran/work_fran/SmoothAttention/data/', train=(subset=="train"), download=True)
            self.dataset.data = self.dataset.data.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        elif self.name == 'mnist':
            self.dataset = torchvision.datasets.MNIST(root='/home/fran/work_fran/SmoothAttention/data/', train=(subset=="train"), download=True)
            self.dataset.data = self.dataset.data.numpy().reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        self.num_bags = num_bags
        self.obj_labels = obj_labels
        self.bag_size = bag_size
        self.use_inst_labels = use_inst_labels

        self.bag_data = None
        self.bag_labels = None
        self.inst_labels = None

        np.random.seed(seed)
        self.create_bags()
    
    def create_bags(self):
        pos_idx = np.where(np.isin(self.dataset.targets, self.obj_labels))[0]
        np.random.shuffle(pos_idx)
        neg_idx = np.where(~np.isin(self.dataset.targets, self.obj_labels))[0]
        np.random.shuffle(neg_idx)

        num_pos_bags = self.num_bags // 2
        num_neg_bags = self.num_bags - num_pos_bags

        pos_idx_queue = deque(pos_idx)
        neg_idx_queue = deque(neg_idx)

        self.bag_data = []
        self.bag_labels = []
        self.inst_labels = []
        self.L_mat = []
        for i in range(num_pos_bags):
            bag = []
            y_labels = []
            num_positives = np.random.randint(1, self.bag_size//2)
            num_negatives = self.bag_size - num_positives
            for _ in range(num_positives):
                a = pos_idx_queue.pop()
                bag.append(self.dataset.data[a])
                y_labels.append(self.dataset.targets[a])
                pos_idx_queue.appendleft(a)
            for _ in range(num_negatives):
                a = neg_idx_queue.pop()
                bag.append(self.dataset.data[a])
                y_labels.append(self.dataset.targets[a])
                neg_idx_queue.appendleft(a)

            idx_sort = np.argsort(y_labels)
            bag = np.stack(bag)[idx_sort]
            y_labels = np.array(y_labels)[idx_sort]
            y_labels = np.where(np.isin(y_labels, self.obj_labels), 1, 0)
            bag_label = np.max(y_labels)

            self.bag_data.append(bag)
            self.bag_labels.append(bag_label)
            self.inst_labels.append(y_labels)
            self.L_mat.append(self.build_L_mat(y_labels).astype(np.float32))

        for i in range(num_neg_bags):
            bag = []
            y_labels = []
            for _ in range(self.bag_size):
                a = neg_idx_queue.pop()
                bag.append(self.dataset.data[a])
                y_labels.append(self.dataset.targets[a])
                neg_idx_queue.appendleft(a)

            idx_sort = np.argsort(y_labels)
            bag = np.stack(bag)[idx_sort]
            y_labels = np.array(y_labels)[idx_sort]
            y_labels = np.zeros_like(y_labels)
            bag_label = 0

            self.bag_data.append(bag)
            self.bag_labels.append(bag_label)
            self.inst_labels.append(y_labels)
            self.L_mat.append(self.build_L_mat(y_labels).astype(np.float32))
    
    def __len__(self):      
        return len(self.bag_data)
    
    def __getitem__(self, index):
        return torch.from_numpy(self.bag_data[index]).float(), torch.tensor(self.bag_labels[index]).float(), torch.from_numpy(self.inst_labels[index]).float(), torch.from_numpy(self.L_mat[index]).float()

    def build_L_mat(self, y_labels, num_neighbors=1):
        n = len(y_labels)
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] = 1
            for j in range(1, num_neighbors+1):
                if i-j >= 0:
                    if self.use_inst_labels:
                        if y_labels[i-j] == y_labels[i]:
                            A[i, i-j] = 1
                            A[i-j, i] = 1
                    else:
                        A[i, i-j] = (num_neighbors - j + 1) / num_neighbors
                        A[i-j, i] = (num_neighbors - j + 1) / num_neighbors
                if i+j < n:
                    if self.use_inst_labels:
                        if y_labels[i+j] == y_labels[i]:
                            A[i, i+j] = 1
                            A[i+j, i] = 1
                    else:
                        A[i, i+j] = (num_neighbors - j + 1) / num_neighbors
                        A[i+j, i] = (num_neighbors - j + 1) / num_neighbors
        return csgraph.laplacian(A, normed=True)
        #return A