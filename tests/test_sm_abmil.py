import sys
sys.path.append('/work/work_fran/torchmil')

import torch
import unittest
from torchmil.models.sm_abmil import SmABMIL

class TestSmABMIL(unittest.TestCase):

    def setUp(self):
        self.input_shape = (512,)
        self.model = SmABMIL(input_shape=self.input_shape)
        self.batch_size = 2
        self.bag_size = 5
        self.X = torch.randn(self.batch_size, self.bag_size, *self.input_shape)
        self.adj_mat = torch.randn(self.batch_size, self.bag_size, self.bag_size)
        self.mask = torch.ones(self.batch_size, self.bag_size)
        self.Y_true = torch.randint(0, 2, (self.batch_size,))

    def test_forward(self):
        Y_logits_pred = self.model.forward(self.X, self.adj_mat, self.mask)
        self.assertEqual(Y_logits_pred.shape, (self.batch_size,))

    def test_forward_with_attention(self):
        Y_logits_pred, att = self.model.forward(self.X, self.adj_mat, self.mask, return_att=True)
        self.assertEqual(Y_logits_pred.shape, (self.batch_size,))
        self.assertEqual(att.shape, (self.batch_size, self.bag_size))

    def test_compute_loss(self):
        Y_logits_pred, loss_dict = self.model.compute_loss(self.Y_true, self.X, self.adj_mat, self.mask)
        self.assertEqual(Y_logits_pred.shape, (self.batch_size,))
        self.assertIn('BCEWithLogitsLoss', loss_dict)
        self.assertIsInstance(loss_dict['BCEWithLogitsLoss'], torch.Tensor)

    def test_predict(self):
        T_logits_pred = self.model.predict(self.X, self.adj_mat, self.mask, return_y_pred=False)
        self.assertEqual(T_logits_pred.shape, (self.batch_size,))

    def test_predict_with_instance_labels(self):
        T_logits_pred, y_pred = self.model.predict(self.X, self.adj_mat, self.mask, return_y_pred=True)
        self.assertEqual(T_logits_pred.shape, (self.batch_size,))
        self.assertEqual(y_pred.shape, (self.batch_size, self.bag_size))

if __name__ == '__main__':
    unittest.main()