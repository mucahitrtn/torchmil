import os

import numpy as np

import datasets

_CITATION = """
"""

_DESCRIPTION = """\
RSNA_ICH_MIL dataset, released as part of the torchmil library (https://torchmil.github.io/).
"""

_HOMEPAGE = "https://torchmil.github.io/"

_LICENSE = "Apache License 2.0"

# MAIN_PATH = './dataset/patches_512/'
MAIN_PATH = '/home/fran/data/datasets/RSNA_ICH'

# Data urls
_DATA_URLS = {
    'labels': f'{MAIN_PATH}/labels.tar.gz',
    'inst_labels': f'{MAIN_PATH}/slice_labels.tar.gz',
}

for features in ['resnet50', 'vit_b_32', 'resnet18']:
    _DATA_URLS[f'features_{features}'] = f'{MAIN_PATH}/features/features_{features}.tar.gz'

# Features dimensions
_FEATURES_DIMS = {
    'resnet50': 2048,
    'vit_b_32': 768,
    'resnet18': 512,
}

class RSNA_ICH_MIL(datasets.GeneratorBasedBuilder):
    """
    RSNA_ICH_MIL dataset, adapted for MIL from the RSNA Intracranial Hemorrhage Detection Challenge (https://www.rsna.org/rsnai/ai-image-challenge/rsna-intracranial-hemorrhage-detection-challenge-2019).
    This dataset is released as part of the torchmil library (https://torchmil.github.io/).

    To create this dataset, we preprocessed the slices in the DCOM files using the code in https://github.com/YunanWu2168/SA-MIL/blob/master/SA_MIL_preprocessing.ipynb. 
    Then, we extracted features from these slices using different models and saved them in .npy files.

    The available features are:
    - "resnet50": a ResNet50 model pre-trained on ImageNet (https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html).
    - "vit_b_32": a Vision Transformer model pre-trained on ImageNet (https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_32.html).
    - "resnet18": a ResNet18 model pre-trained on ImageNet (https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html).
    
    To choose the features, use the corresponding name in the config argument.

    The dataset returns a dictionary containing the following fields:
    - "bag_name": the name of the WSI.
    - "features": a 2D array containing the features of the patches in the bag.
    - "label": the label of the bag (0 or 1).
    - "inst_labels": a 2D array containing the labels of the patches in the bag.
    """

    def _info(self):
        features_dim = _FEATURES_DIMS[self.config.name]
        features = datasets.Features(
            {
                'bag_name': datasets.Value("string"),
                'features': datasets.Array2D(shape=(None, features_dim), dtype="float32"),
                'label': datasets.ClassLabel(names=[0, 1]),
                'inst_labels': datasets.Array2D(shape=(None, 1), dtype="int32"),
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

        print('Downloading and extracting features...')
        features_path = dl_manager.download_and_extract([_DATA_URLS[f'features_{self.config.name}']])
        print('Features path:', features_path)

        print('Downloading and extracting labels...')
        labels_path = dl_manager.download_and_extract(_DATA_URLS['labels'])

        print('Downloading and extracting patch labels...')
        inst_labels_path = dl_manager.download_and_extract(_DATA_URLS['inst_labels'])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'features_path' : features_path,
                    'labels_path': labels_path,
                    'inst_labels_path': inst_labels_path,
                },
            )
        ]

    def _generate_examples(self, features_path, labels_path, inst_labels_path, coords_path):

        bag_names = os.listdir(features_path)
        for bag_name in bag_names:
            print(os.path.join(features_path, bag_name))
            features = np.load(os.path.join(features_path, bag_name)).astype(np.float32)
            label = np.load(os.path.join(labels_path, bag_name)).item()
            inst_labels = np.load(os.path.join(inst_labels_path, bag_name)).astype(np.int32)

            yield bag_name, {
                'bag_name': bag_name,
                'features': features,
                'label': label,
                'inst_labels': inst_labels,               
            }