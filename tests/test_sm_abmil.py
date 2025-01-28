import sys
sys.path.append('/work/work_fran/torchmil')

import torch
import unittest
from torchmil.models.sm_abmil import SmABMIL

class TestSmABMIL(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.bag_size = 10
        self.feat_dim = 16
        self.model = SmABMIL(in_shape=None)

        self.X = torch.randn(self.batch_size, self.bag_size, self.feat_dim)
        self.adj_mat = torch.randn(self.batch_size, self.bag_size, self.bag_size)
        self.mask = torch.ones(self.batch_size, self.bag_size)
        self.Y_true = torch.randint(0, 2, (self.batch_size,)).float()

    def test_forward(self):
        bag_pred = self.model.forward(self.X, self.adj_mat, self.mask)
        self.assertEqual(bag_pred.shape, (self.batch_size,))

    def test_forward_with_attention(self):
        bag_pred, att = self.model.forward(self.X, self.adj_mat, self.mask, return_att=True)
        self.assertEqual(bag_pred.shape, (self.batch_size,))
        self.assertEqual(att.shape, (self.batch_size, self.bag_size))

    def test_compute_loss(self):
        bag_pred, loss_dict = self.model.compute_loss(self.Y_true, self.X, self.adj_mat, self.mask)
        self.assertEqual(bag_pred.shape, (self.batch_size,))
        self.assertIn('BCEWithLogitsLoss', loss_dict)
        self.assertIsInstance(loss_dict['BCEWithLogitsLoss'], torch.Tensor)

    def test_predict(self):
        bag_pred = self.model.predict(self.X, self.adj_mat, self.mask, return_inst_pred=False)
        self.assertEqual(bag_pred.shape, (self.batch_size,))

    def test_predict_with_instance_predictions(self):
        bag_pred, inst_pred = self.model.predict(self.X, self.adj_mat, self.mask, return_inst_pred=True)
        self.assertEqual(bag_pred.shape, (self.batch_size,))
        self.assertEqual(inst_pred.shape, (self.batch_size, self.bag_size))

if __name__ == '__main__':
    unittest.main()