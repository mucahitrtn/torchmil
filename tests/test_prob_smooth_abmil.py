import sys
sys.path.append('/work/work_fran/torchmil')

import unittest
import torch
from torchmil.models import ProbSmoothABMIL

class TestProbSmoothABMIL(unittest.TestCase):

    def setUp(self):
        self.input_shape = (512,)
        self.model = ProbSmoothABMIL(input_shape=self.input_shape)
        self.batch_size = 2
        self.bag_size = 5
        self.X = torch.randn(self.batch_size, self.bag_size, *self.input_shape)
        self.mask = torch.ones(self.batch_size, self.bag_size)
        self.adj_mat = torch.ones(self.batch_size, self.bag_size, self.bag_size)
        self.Y_true = torch.randint(0, 2, (self.batch_size,))

    def test_forward(self):
        Y_logits_pred = self.model.forward(self.X, self.mask)
        self.assertEqual(Y_logits_pred.shape, (self.batch_size,))

    def test_forward_with_attention(self):
        Y_logits_pred, att = self.model.forward(self.X, self.mask, return_att=True)
        self.assertEqual(Y_logits_pred.shape, (self.batch_size,))
        self.assertEqual(att.shape, (self.batch_size, self.bag_size))

    def test_forward_with_kl_div(self):
        Y_logits_pred, kl_div = self.model.forward(self.X, self.mask, self.adj_mat, return_kl_div=True)
        self.assertEqual(Y_logits_pred.shape, (self.batch_size,))
        self.assertEqual(kl_div.shape, ())

    def test_compute_loss(self):
        Y_logits_pred, loss_dict = self.model.compute_loss(self.Y_true, self.X, self.adj_mat, self.mask)
        self.assertEqual(Y_logits_pred.shape, (self.batch_size,))
        self.assertIn('BCEWithLogitsLoss', loss_dict)
        self.assertIn('KLDiv', loss_dict)

    def test_predict(self):
        Y_logits_pred, att_val = self.model.predict(self.X, self.mask)
        self.assertEqual(Y_logits_pred.shape, (self.batch_size,))
        self.assertEqual(att_val.shape, (self.batch_size, self.bag_size))

if __name__ == '__main__':
    unittest.main()