import torch

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_list):
        super(ListDataset, self).__init__()
        self.imgs_list = imgs_list

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        return torch.from_numpy(self.imgs_list[idx])