import sys
sys.path.append('/work/work_fran/torchmil')


import unittest
import numpy as np
from torchmil.datasets import ToyMIL

class TestToyMIL(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.data = np.random.rand(100, 10)  # 100 instances, 10 features each
        self.labels = np.random.randint(0, 2, 100)  # Binary labels
        self.num_bags = 10
        self.obj_labels = [1]
        self.bag_size = 5
        self.pos_class_prob = 0.5
        self.seed = 42

        self.dataset = ToyMIL(
            data=self.data,
            labels=self.labels,
            num_bags=self.num_bags,
            obj_labels=self.obj_labels,
            bag_size=self.bag_size,
            pos_class_prob=self.pos_class_prob,
            seed=self.seed
        )

    def test_len(self):
        self.assertEqual(len(self.dataset), self.num_bags)

    def test_getitem(self):
        for i in range(len(self.dataset)):
            bag = self.dataset[i]
            self.assertIn('data', bag)
            self.assertIn('label', bag)
            self.assertIn('inst_labels', bag)
            self.assertEqual(bag['data'].shape[0], self.bag_size)
            self.assertEqual(bag['inst_labels'].shape[0], self.bag_size)
            self.assertTrue(bag['label'].item() in [0, 1])

    def test_positive_bags(self):
        num_positive_bags = sum([self.dataset[i]['label'].item() for i in range(len(self.dataset))])
        expected_positive_bags = int(self.num_bags * self.pos_class_prob)
        self.assertEqual(num_positive_bags, expected_positive_bags)

    def test_negative_bags(self):
        num_negative_bags = len(self.dataset) - sum([self.dataset[i]['label'].item() for i in range(len(self.dataset))])
        expected_negative_bags = self.num_bags - int(self.num_bags * self.pos_class_prob)
        self.assertEqual(num_negative_bags, expected_negative_bags)

if __name__ == '__main__':
    unittest.main()