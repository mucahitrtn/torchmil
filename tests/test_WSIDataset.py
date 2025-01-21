import sys
sys.path.append('/work/work_fran/torchmil')

import unittest
import numpy as np
import os
import shutil
from torchmil.datasets.wsi_dataset import WSIDataset, build_adj_WSI
from tensordict import TensorDict


class TestWSIDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = 'test_data'
        cls.labels_path = 'test_labels'
        cls.inst_labels_path = 'test_inst_labels'
        cls.coords_path = 'test_coords'

        os.makedirs(cls.data_path, exist_ok=True)
        os.makedirs(cls.labels_path, exist_ok=True)
        os.makedirs(cls.inst_labels_path, exist_ok=True)
        os.makedirs(cls.coords_path, exist_ok=True)

        # Create dummy data
        for i in range(5):
            data = np.random.rand(10, 2048)
            label = np.random.randint(0, 2, size=(1,))
            inst_labels = np.random.randint(0, 2, size=(10,))
            coords = np.random.rand(10, 2) * 1000

            np.save(os.path.join(cls.data_path, f'bag_{i}.npy'), data)
            np.save(os.path.join(cls.labels_path, f'bag_{i}.npy'), label)
            np.save(os.path.join(cls.inst_labels_path, f'bag_{i}.npy'), inst_labels)
            np.save(os.path.join(cls.coords_path, f'bag_{i}.npy'), coords)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.data_path)
        shutil.rmtree(cls.labels_path)
        shutil.rmtree(cls.inst_labels_path)
        shutil.rmtree(cls.coords_path)

    def test_len(self):
        dataset = WSIDataset(
            data_path=self.data_path,
            labels_path=self.labels_path,
            inst_labels_path=self.inst_labels_path,
            coords_path=self.coords_path
        )
        self.assertEqual(len(dataset), 5)

    def test_getitem(self):
        dataset = WSIDataset(
            data_path=self.data_path,
            labels_path=self.labels_path,
            inst_labels_path=self.inst_labels_path,
            coords_path=self.coords_path
        )
        item = dataset[0]
        self.assertIsInstance(item, TensorDict)
        self.assertIn('data', item)
        self.assertIn('label', item)
        self.assertIn('inst_labels', item)
        self.assertIn('coords', item)
        self.assertIn('edge_index', item)
        self.assertIn('edge_weight', item)

        self.assertEqual(item['inst_labels'].shape[0], item['data'].shape[0])
        self.assertEqual(item['coords'].shape[0], item['data'].shape[0])
        self.assertTrue(item['label'].item() in [0, 1])

    def test_build_adj_WSI(self):
        coords = np.random.rand(10, 2) * 1000
        feat = np.random.rand(10, 2048)
        edge_index, edge_weight = build_adj_WSI(coords, feat, patch_size=512, add_self_loops=True)
        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(edge_index.shape[1], len(edge_weight))
        self.assertTrue(np.all(edge_weight > 0))

if __name__ == '__main__':
    unittest.main()