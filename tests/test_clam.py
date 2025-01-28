import sys
sys.path.append('/work/work_fran/torchmil')

import unittest
import torch
from torchmil.models.clam.clam import CLAM_SB

class TestCLAMSB(unittest.TestCase):
    def setUp(self):
        self.input_shape = (256,)
        self.att_dim = 128
        self.att_act = 'tanh'
        self.dropout = 0.5
        self.k_sample = 10
        self.inst_loss_name = 'svm'
        self.feat_ext = torch.nn.Identity()
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.model = CLAM_SB(
            input_shape=self.input_shape,
            att_dim=self.att_dim,
            att_act=self.att_act,
            dropout=self.dropout,
            k_sample=self.k_sample,
            inst_loss_name=self.inst_loss_name,
            feat_ext=self.feat_ext,
            criterion=self.criterion
        )

    def test_create_positive_targets(self):
        length = 10
        device = torch.device('cpu')
        pos_targets = self.model.create_positive_targets(length, device)
        expected_shape = (length,)
        expected_values = torch.full((length,), 1, device=device).long()
        self.assertEqual(pos_targets.shape, expected_shape)
        self.assertTrue(torch.all(torch.eq(pos_targets, expected_values)))

    def test_create_negative_targets(self):
        length = 10
        device = torch.device('cpu')
        neg_targets = self.model.create_negative_targets(length, device)
        expected_shape = (length,)
        expected_values = torch.full((length,), 0, device=device).long()
        self.assertEqual(neg_targets.shape, expected_shape)
        self.assertTrue(torch.all(torch.eq(neg_targets, expected_values)))

    def test_inst_eval(self):
        att = torch.randn(10)
        emb = torch.randn(10, *self.input_shape)
        classifier = torch.nn.Linear(*self.input_shape, 2)
        instance_loss, all_preds, all_targets = self.model.inst_eval(att, emb, classifier)
        self.assertIsInstance(instance_loss, torch.Tensor)
        self.assertIsInstance(all_preds, torch.Tensor)
        self.assertIsInstance(all_targets, torch.Tensor)

    def test_inst_eval_out(self):
        att = torch.randn(10)
        emb = torch.randn(10, *self.input_shape)
        classifier = torch.nn.Linear(*self.input_shape, 2)
        instance_loss, p_preds, p_targets = self.model.inst_eval_out(att, emb, classifier)
        self.assertIsInstance(instance_loss, torch.Tensor)
        self.assertIsInstance(p_preds, torch.Tensor)
        self.assertIsInstance(p_targets, torch.Tensor)

    def test_compute_inst_loss(self):
        att = torch.randn(4, 10)
        emb = torch.randn(4, 10, *self.input_shape)
        labels = torch.randint(0, 2, (10,))
        inst_loss = self.model.compute_inst_loss(att, emb, labels)
        self.assertIsInstance(inst_loss, torch.Tensor)

    def test_forward(self):
        X = torch.randn(5, 10, *self.input_shape)
        bag_pred = self.model.forward(X)
        self.assertIsInstance(bag_pred, torch.Tensor)

    def test_compute_loss(self):
        labels = torch.randint(0, 2, (5,))
        X = torch.randn(5, 10, *self.input_shape)
        mask = torch.ones(5, 10)
        bag_pred, loss_dict = self.model.compute_loss(labels, X, mask)
        self.assertIsInstance(bag_pred, torch.Tensor)
        self.assertIsInstance(loss_dict, dict)

    def test_predict(self):
        X = torch.randn(5, 10, *self.input_shape)
        bag_pred, inst_pred = self.model.predict(X)
        self.assertIsInstance(bag_pred, torch.Tensor)
        self.assertIsInstance(inst_pred, torch.Tensor)

if __name__ == '__main__':
    unittest.main()