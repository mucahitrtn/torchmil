import sys
sys.path.append('/work/work_fran/torchmil')

import unittest
from torchmil.datasets.camelyon16 import Camelyon16

class TestCamelyon16(unittest.TestCase):
    def setUp(self):
        self.dataset = Camelyon16()

    def test_items(self):
        for i in range(len(self.dataset)):
            bag_name = self.dataset.bag_names[i]
            item = self.dataset[i]
            print(bag_name)
            for k in item.keys():
                print('\t', k, item[k].shape)

if __name__ == '__main__':
    unittest.main()