import sys
sys.path.append('/work/work_fran/torchmil')

import unittest
import torch
from torchmil.models.abmil import ABMIL

class TestABMIL(unittest.TestCase):

    def setUp(self):
        self.input_shape = (512,)
        self.att_dim = 128
        self.att_act = 'tanh'
        self.feat_ext = torch.nn.Identity()
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.model = ABMIL(self.input_shape, self.att_dim, self.att_act, self.feat_ext, self.criterion)

    def test_abmil_initialization(self):
        self.assertEqual(self.model.input_shape, self.input_shape)
        self.assertEqual(self.model.criterion, self.criterion)
        self.assertIsInstance(self.model.feat_ext, torch.nn.Identity)
        self.assertEqual(self.model.pool.att_dim, self.att_dim)
        self.assertEqual(self.model.pool.act, self.att_act)
        self.assertIsInstance(self.model.classifier, torch.nn.Linear)

    def test_abmil_forward(self):
        batch_size = 2
        bag_size = 3
        feat_dim = 4
        input_shape = (bag_size, feat_dim)
        model = ABMIL(input_shape, self.att_dim, self.att_act, self.feat_ext, self.criterion)
        X = torch.randn(batch_size, bag_size, feat_dim)
        mask = torch.ones(batch_size, bag_size)

        Y_logits_pred = model.forward(X, mask, return_att=False)
        self.assertEqual(Y_logits_pred.shape, (batch_size,))

        Y_logits_pred, att = model.forward(X, mask, return_att=True)
        self.assertEqual(Y_logits_pred.shape, (batch_size,))
        self.assertEqual(att.shape, (batch_size, bag_size))

    def test_abmil_compute_loss(self):
        batch_size = 2
        bag_size = 3
        feat_dim = 4
        input_shape = (bag_size, feat_dim)
        model = ABMIL(input_shape, self.att_dim, self.att_act, self.feat_ext, self.criterion)
        X = torch.randn(batch_size, bag_size, feat_dim)
        mask = torch.ones(batch_size, bag_size)
        Y_true = torch.randint(0, 2, (batch_size,))

        Y_logits_pred, loss_dict = model.compute_loss(Y_true, X, mask)
        self.assertEqual(Y_logits_pred.shape, (batch_size,))
        self.assertIn('BCEWithLogitsLoss', loss_dict)
        self.assertEqual(loss_dict['BCEWithLogitsLoss'].shape, ())

    def test_abmil_predict(self):
        batch_size = 2
        bag_size = 3
        feat_dim = 4
        input_shape = (bag_size, feat_dim)
        model = ABMIL(input_shape, self.att_dim, self.att_act, self.feat_ext, self.criterion)
        X = torch.randn(batch_size, bag_size, feat_dim)
        mask = torch.ones(batch_size, bag_size)

        Y_logits_pred, f = model.predict(X, mask, return_y_pred=True)
        self.assertEqual(Y_logits_pred.shape, (batch_size,))
        self.assertEqual(f.shape, (batch_size, bag_size))

        Y_logits_pred = model.predict(X, mask, return_y_pred=False)
        self.assertEqual(Y_logits_pred.shape, (batch_size,))

if __name__ == "__main__":
    unittest.main()
