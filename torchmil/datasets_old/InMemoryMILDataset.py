import torch
import numpy as np

from .MILDataset import MILDataset

class InMemoryMILDataset(MILDataset):
    def __init__(self, 
                 data_path : str, 
                 csv_path : str,
                 *args,
                 **kwargs
                 ):
        super(InMemoryMILDataset, self).__init__(data_path, csv_path)
        self.loaded_dict = { bag_name: False for bag_name in self.bag_names }
        for bag_name in self.bag_names:
            self.data_dict[bag_name]['bag_data'] = None
            
        # Loaded dict: { bag_name: True/False }
    
    def _compute_data_shape(self):
        tmp = self._loader(self.data_dict[self.bag_names[0]]['inst_paths'][0])
        return tmp.shape

    def _get_bag_feat(self, bag_name):

        if not self.loaded_dict[bag_name]:
            bag_feat = self._load_bag_feat(bag_name)
            self.data_dict[bag_name]['bag_feat'] = bag_feat
            self.loaded_dict[bag_name] = True
        else:
            bag_feat = self.data_dict[bag_name]['bag_feat']
        
        return bag_feat