import os
import torch
import cv2
import numpy as np

from tqdm import tqdm

class CustomDatasetFolder(torch.utils.data.Dataset):
    def __init__(self, root, files_list=None, load_at_init=False, resize_size=None):
        super().__init__()
        self.extensions = ('.jpg', '.jpeg', '.png', '.npy')
        self.load_at_init = load_at_init
        self.files_list = files_list
        self.root = root
        self.resize_size = resize_size
        self.path_list = self._make_dataset(root, files_list)
        if self.load_at_init:
            self.img_list = self._load_imgs()
        else:
            self.img_list = None
    
    def _compute_data_shape(self):
        tmp = self.__getitem__(0)[0]
        return tmp.shape
    
    def _load(self, path):
        if path.lower().endswith('.npy'):
            sample = np.load(path, allow_pickle=True).astype(np.float32)
        elif path.lower().endswith(self.extensions):
            sample = cv2.imread(path)
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB) # (H, W, C)
            # sample = np.transpose(sample, (2, 0, 1)) # (C, H, W)
        else:
            raise NotImplementedError
        
        if self.resize_size is not None:
            sample = cv2.resize(sample, (self.resize_size, self.resize_size))

        return sample

    def _load_imgs(self):
        pbar = tqdm(self.path_list, total=len(self.path_list))
        pbar.set_description_str("[CustomDatasetFolder] Loading files...")
        img_list = []
        for path in pbar:
            pbar.set_postfix_str(path)
            img = self._load(path)
            img_list.append(img)
        return img_list
    
    def _is_valid_file(self, path):
        extension_correct = path.lower().endswith(self.extensions)
        # file_exists = os.path.isfile(path)
        return extension_correct
    
    def _make_dataset(self, dir, files_list=None):
        if files_list is None:
            files_list = os.listdir(dir)
        # else:
        #     existing_files = os.listdir(dir)
        #     files_list = set(files_list).intersection(set(existing_files))

        pbar = tqdm(files_list, total=len(files_list))
        pbar.set_description_str("[CustomDatasetFolder] Scanning files...")
        path_list = [ os.path.join(dir, file) for file in pbar if self._is_valid_file(os.path.join(dir, file)) ]
        print(f"[CustomDatasetFolder] Found {len(path_list)} images.")
        # for file in pbar:
        #     path = os.path.join(dir, file)
        #     pbar.set_postfix_str(path)
        #     if os.path.isdir(path):
        #         path_list = path_list + self._make_dataset(path)
        #     elif self._is_valid_file(path):
        #         path_list.append(path)
        return path_list
    
    def __getitem__(self, index):
        path = self.path_list[index]
        if self.load_at_init:
            sample = self.img_list[index] # (H, W, C)
        else:
            sample = self._load(path) # (H, W, C)

        sample = torch.from_numpy(sample).permute(2, 0, 1).float() # (C, H, W)        
        return (sample, path)

    def __len__(self):
        return len(self.path_list)