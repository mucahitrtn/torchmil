import pandas as pd

from .WSIDataset import WSIDataset

class CAMELYON16Dataset(WSIDataset):

    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'dataset_name'):
            self.dataset_name = 'CAMELYON16Dataset'
        super(CAMELYON16Dataset, self).__init__(*args, **kwargs)
    
    def _read_csv(self):
        df = pd.read_csv(self.csv_path)

        bag_names_list = df['wsi_name'].apply(lambda x: x.split('.')[0]).values
        bag_labels_list = df['wsi_label'].astype(int).values

        return bag_names_list, bag_labels_list