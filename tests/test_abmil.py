import sys
sys.path.append('/work/work_fran/torchmil')

import torch
import unittest
from torchmil.models.abmil import ABMIL

class TestABMIL(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.bag_size = 10
        self.feat_dim = 16
        self.model = ABMIL(in_shape=None)

    def test_forward(self):
        X = torch.randn(self.batch_size, self.bag_size, self.feat_dim)
        mask = torch.ones(self.batch_size, self.bag_size)
        bag_pred = self.model.forward(X, mask)
        self.assertEqual(bag_pred.shape, (self.batch_size,))

    def test_forward_with_attention(self):
        X = torch.randn(self.batch_size, self.bag_size, self.feat_dim)
        mask = torch.ones(self.batch_size, self.bag_size)
        bag_pred, att = self.model.forward(X, mask, return_att=True)
        self.assertEqual(bag_pred.shape, (self.batch_size,))
        self.assertEqual(att.shape, (self.batch_size, self.bag_size))

    def test_compute_loss(self):
        X = torch.randn(self.batch_size, self.bag_size, self.feat_dim)
        labels = torch.randint(0, 2, (self.batch_size,))
        mask = torch.ones(self.batch_size, self.bag_size)
        bag_pred, loss_dict = self.model.compute_loss(labels, X, mask)
        self.assertEqual(bag_pred.shape, (self.batch_size,))
        self.assertIn('BCEWithLogitsLoss', loss_dict)
        self.assertIsInstance(loss_dict['BCEWithLogitsLoss'], torch.Tensor)

    def test_predict(self):
        X = torch.randn(self.batch_size, self.bag_size, self.feat_dim)
        mask = torch.ones(self.batch_size, self.bag_size)
        bag_pred, inst_pred = self.model.predict(X, mask, return_inst_pred=True)
        self.assertEqual(bag_pred.shape, (self.batch_size,))
        self.assertEqual(inst_pred.shape, (self.batch_size, self.bag_size))
    
if __name__ == '__main__':
    unittest.main()