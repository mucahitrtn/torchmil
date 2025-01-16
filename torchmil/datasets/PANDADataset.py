
import pandas as pd

from .WSIDataset import WSIDataset

class PANDADataset(WSIDataset):

    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'dataset_name'):
            self.dataset_name = 'PANDADataset'
        super(PANDADataset, self).__init__(*args, **kwargs)
    
    def _read_csv(self):
        df = pd.read_csv(self.csv_path)

        bag_names_list = df['image_id'].values
        bag_labels_list = df['isup_grade'].astype(int).apply(lambda x : 0 if x == 0 else 1).values

        return bag_names_list, bag_labels_list