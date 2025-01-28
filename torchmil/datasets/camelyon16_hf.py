import os

import numpy as np

import datasets

_CITATION = """
"""

_DESCRIPTION = """\
Camelyon16_MIL dataset, released as part of the torchmil library (https://torchmil.github.io/).
"""

_HOMEPAGE = "https://torchmil.github.io/"

_LICENSE = "Apache License 2.0"

# MAIN_PATH = './dataset/patches_512/'
MAIN_PATH = '/home/fran/data/datasets/CAMELYON16/patches_512_preset/'

# Data urls
_DATA_URLS = {
    'coords': f'{MAIN_PATH}/coords.tar.gz',
    'labels': f'{MAIN_PATH}/labels.tar.gz',
    'inst_labels': f'{MAIN_PATH}/patch_labels.tar.gz',
}

for features in ['resnet50', 'vit_b_32', 'resnet50_bt', 'UNI']:
    _DATA_URLS[f'features_{features}'] = f'{MAIN_PATH}/features/features_{features}.tar.gz'

# Features dimensions
_FEATURES_DIMS = {
    'resnet50': 2048,
    'vit_b_32': 768,
    'resnet50_bt': 2048,
    'UNI': 2048
}

class Camelyon16_MIL(datasets.GeneratorBasedBuilder):
    """

    Camelyon16_MIL dataset, adapted for MIL from the CAMELYON16 challenge (https://camelyon16.grand-challenge.org/). 
    This dataset is released as part of the torchmil library (https://torchmil.github.io/).

    To create this dataset, we extracted patches of size 512x512 from the original WSIs using the CLAM tool (https://github.com/mahmoodlab/CLAM). 
    Then, we extracted features from these patches using different models and saved them in .npy files.

    The available features are:
    - "resnet50": a ResNet50 model pre-trained on ImageNet (https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html).
    - "vit_b_32": a Vision Transformer model pre-trained on ImageNet (https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_32.html).
    - "resnet50_bt": a ResNet50 model pre-trained using the Barlow Twins method on a huge dataset of WSI patches (https://lunit-io.github.io/research/publications/pathology_ssl/).
    - "UNI": the UNI foundation model (https://www.nature.com/articles/s41591-024-02857-3).

    To choose the features, use the corresponding name in the config argument.

    The dataset returns a dictionary containing the following fields:
    - "bag_name": the name of the WSI.
    - "features": a 2D array containing the features of the patches in the bag.
    - "label": the label of the bag (0 or 1).
    - "inst_labels": a 2D array containing the labels of the patches in the bag.
    - "coords": a 2D array containing the coordinates of the patches in the bag.
    """

    def __init__(self, *args, verbose=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose

    def _info(self):
        features_dim = _FEATURES_DIMS[self.config.name]
        features = datasets.Features(
            {
                'bag_name': datasets.Value("string"),
                'features': datasets.Array2D(shape=(None, features_dim), dtype="float32"),
                'label': datasets.ClassLabel(names=[0, 1]),
                'inst_labels': datasets.Array2D(shape=(None, 1), dtype="int32"),
                'coords': datasets.Array2D(shape=(None, 2), dtype="int32")   
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        if self.verbose:
            print('Downloading and extracting features...')
        features_path = dl_manager.download_and_extract(_DATA_URLS[f'features_{self.config.name}'])
        print('Features path:', features_path)

        if self.verbose:
            print('Downloading and extracting labels...')
        labels_path = dl_manager.download_and_extract(_DATA_URLS['labels'])

        if self.verbose:
            print('Downloading and extracting patch labels...')
        inst_labels_path = dl_manager.download_and_extract(_DATA_URLS['inst_labels'])

        if self.verbose:
            print('Downloading and extracting coords...')
        coords_path = dl_manager.download_and_extract(_DATA_URLS['coords'])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'features_path' : features_path,
                    'labels_path': labels_path,
                    'inst_labels_path': inst_labels_path,
                    'coords_path': coords_path       
                },
            )
        ]

    def _generate_examples(self, features_path, labels_path, inst_labels_path, coords_path):

        bag_names = os.listdir(features_path)
        for bag_name in bag_names:
            features = np.load(os.path.join(features_path, bag_name)).astype(np.float32)
            label = np.load(os.path.join(labels_path, bag_name)).item()
            inst_labels = np.load(os.path.join(inst_labels_path, bag_name)).astype(np.int32)
            coords = np.load(os.path.join(coords_path, bag_name)).astype(np.int32)

            yield bag_name, {
                'bag_name': bag_name,
                'features': features,
                'label': label,
                'inst_labels': inst_labels,
                'coords': coords
            }