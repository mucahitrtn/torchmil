import sys

print(sys.path)
import torch
import pytest
from torch import nn
from torch.testing import assert_close

from torchmil.models import SmABMIL


# Fixtures for common setup
@pytest.fixture
def sample_data():
    # Returns a tuple of (X, adj, Y, mask)
    X = torch.randn(2, 3, 5)  # batch_size, bag_size, feat_dim
    adj = torch.randn(2, 3, 3)  # batch_size, bag_size, bag_size
    Y = torch.randint(0, 2, (2,))  # batch_size
    mask = torch.randint(0, 2, (2, 3)).bool()  # batch_size, bag_size
    return X, adj, Y, mask


@pytest.fixture
def smabmil_model():
    # Returns an instance of the SmABMIL model with default parameters
    return SmABMIL(in_shape=(3, 5))


def test_smabmil_forward_pass(sample_data, smabmil_model):
    # Tests the forward pass of the model with and without mask and return_att
    X, adj, _, mask = sample_data
    Y_pred = smabmil_model(X, adj, mask)
    assert Y_pred.shape == (2,)

    Y_pred = smabmil_model(X, adj)
    assert Y_pred.shape == (2,)

    Y_pred, att = smabmil_model(X, adj, mask, return_att=True)
    assert Y_pred.shape == (2,)
    assert att.shape == (2, 3)


def test_smabmil_compute_loss(sample_data, smabmil_model):
    # Tests the compute_loss method of the model
    X, adj, Y, mask = sample_data
    Y_pred, loss_dict = smabmil_model.compute_loss(Y, X, adj, mask)
    assert Y_pred.shape == (2,)
    assert "BCEWithLogitsLoss" in loss_dict
    assert loss_dict["BCEWithLogitsLoss"].shape == ()  # loss is a scalar


def test_smabmil_predict(sample_data, smabmil_model):
    # Tests the predict method of the model with and without return_inst_pred
    X, adj, _, mask = sample_data
    Y_pred = smabmil_model.predict(X, adj, mask, return_inst_pred=False)
    assert Y_pred.shape == (2,)

    Y_pred, y_inst_pred = smabmil_model.predict(X, adj, mask, return_inst_pred=True)
    assert Y_pred.shape == (2,)
    assert y_inst_pred.shape == (2, 3)


def test_smabmil_with_feature_extractor(sample_data):
    # Tests the model with a feature extractor
    X, adj, Y, mask = sample_data
    feat_ext = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 7),
    )
    model = SmABMIL(
        in_shape=(3, 5), feat_ext=feat_ext
    )  # in_shape is the shape of the original input
    Y_pred = model(X, adj, mask)
    assert Y_pred.shape == (2,)

    Y_pred, loss_dict = model.compute_loss(Y, X, adj, mask)
    assert Y_pred.shape == (2,)
    assert "BCEWithLogitsLoss" in loss_dict
    assert loss_dict["BCEWithLogitsLoss"].shape == ()

def test_smabmil_different_pooling_params(sample_data):
    # Test different attention and Sm pooling parameters
    X, adj, _, mask = sample_data
    model_relu = SmABMIL(in_shape=(3, 5), att_act="relu")
    model_exact_sm = SmABMIL(in_shape=(3, 5), sm_mode="exact")
    model_trainable_alpha = SmABMIL(in_shape=(3, 5), sm_alpha="trainable")
    model_sm_layers = SmABMIL(in_shape=(3, 5), sm_layers=2)
    model_sm_pre = SmABMIL(in_shape=(3, 5), sm_pre=True)
    model_sm_post = SmABMIL(in_shape=(3, 5), sm_post=True)
    model_spectral_norm = SmABMIL(in_shape=(3, 5), sm_spectral_norm=True)

    Y_pred_relu = model_relu(X, adj, mask)
    Y_pred_exact_sm = model_exact_sm(X, adj, mask)
    Y_pred_trainable_alpha = model_trainable_alpha(X, adj, mask)
    Y_pred_sm_layers = model_sm_layers(X, adj, mask)
    Y_pred_sm_pre = model_sm_pre(X, adj, mask)
    Y_pred_sm_post = model_sm_post(X, adj, mask)
    Y_pred_spectral_norm = model_spectral_norm(X, adj, mask)

    assert Y_pred_relu.shape == (2,)
    assert Y_pred_exact_sm.shape == (2,)
    assert Y_pred_trainable_alpha.shape == (2,)
    assert Y_pred_sm_layers.shape == (2,)
    assert Y_pred_sm_pre.shape == (2,)
    assert Y_pred_sm_post.shape == (2,)
    assert Y_pred_spectral_norm.shape == (2,)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__])
