
import os

from torchmil.datasets import WSIDataset

BASE_DIR = '/home/fran/data/datasets/CAMELYON16/patches_512_preset'

class Camelyon16(WSIDataset):
    def __init__(self):

        super(Camelyon16, self).__init__(
            data_path=os.path.join(BASE_DIR, 'features/features_UNI'),
            labels_path=os.path.join(BASE_DIR, 'labels'),
            inst_labels_path=os.path.join(BASE_DIR, 'patch_labels'),
            coords_path=os.path.join(BASE_DIR, 'coords'),
            patch_size=512
        )