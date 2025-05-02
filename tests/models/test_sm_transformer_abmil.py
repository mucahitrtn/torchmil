import pytest
import torch
from torch.nn import BCEWithLogitsLoss, Identity

from torchmil.nn import SmAttentionPool, SmTransformerEncoder
from torchmil.models.sm_transformer_abmil import SmTransformerABMIL


def test_sm_transformer_abmil_initialization():
    # Define input shape
    in_dim = 256
    in_shape = (in_dim,)

    # Test with default parameters
    model = SmTransformerABMIL(in_shape=in_shape)
    assert isinstance(model, SmTransformerABMIL)
    assert isinstance(model.feat_ext, Identity)
    assert isinstance(model.transformer_encoder, SmTransformerEncoder)
    assert isinstance(model.pool, SmAttentionPool)
    assert isinstance(model.last_layer, torch.nn.Linear)
    assert isinstance(model.criterion, BCEWithLogitsLoss)

    # Test with custom parameters
    class CustomFeatExt(torch.nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.linear = torch.nn.Linear(in_dim, 128)

        def forward(self, x):
            return self.linear(x)
    feat_ext = CustomFeatExt(in_dim)
    pool_att_dim = 64
    pool_act = 'relu'
    pool_sm_mode = 'exact'
    pool_sm_alpha = 0.8
    pool_sm_layers = 2
    pool_sm_steps = 5
    pool_sm_pre = True
    pool_sm_post = True
    pool_sm_spectral_norm = True
    transf_att_dim = 256
    transf_n_layers = 2
    transf_n_heads = 8
    transf_use_mlp = False
    transf_add_self = False
    transf_dropout = 0.2
    transf_sm_alpha = 0.2
    transf_sm_mode = 'approx'
    transf_sm_steps = 8
    criterion = torch.nn.CrossEntropyLoss()

    model = SmTransformerABMIL(
        in_shape=in_shape,
        pool_att_dim=pool_att_dim,
        pool_act=pool_act,
        pool_sm_mode=pool_sm_mode,
        pool_sm_alpha=pool_sm_alpha,
        pool_sm_layers=pool_sm_layers,
        pool_sm_steps=pool_sm_steps,
        pool_sm_pre=pool_sm_pre,
        pool_sm_post=pool_sm_post,
        pool_sm_spectral_norm=pool_sm_spectral_norm,
        feat_ext=feat_ext,
        transf_att_dim=transf_att_dim,
        transf_n_layers=transf_n_layers,
        transf_n_heads=transf_n_heads,
        transf_use_mlp=transf_use_mlp,
        transf_add_self=transf_add_self,
        transf_dropout=transf_dropout,
        transf_sm_alpha=transf_sm_alpha,
        transf_sm_mode=transf_sm_mode,
        transf_sm_steps=transf_sm_steps,
        criterion=criterion,
    )
    assert isinstance(model, SmTransformerABMIL)
    assert isinstance(model.feat_ext, CustomFeatExt)
    assert isinstance(model.transformer_encoder, SmTransformerEncoder)
    assert isinstance(model.pool, SmAttentionPool)
    assert isinstance(model.last_layer, torch.nn.Linear)
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss)



def test_sm_transformer_abmil_forward():
    # Define input shape
    in_dim = 256
    in_shape = (in_dim,)
    # Initialize model
    model = SmTransformerABMIL(in_shape=in_shape)
    # Create dummy input
    batch_size = 2
    bag_size = 4
    # The input X should match the in_shape, so (batch_size, bag_size, C, H, W)
    X = torch.randn(batch_size, bag_size, *in_shape)
    adj = torch.randn(batch_size, bag_size, bag_size)
    mask = torch.randint(0, 2, (batch_size, bag_size)).bool()

    # Test forward pass without attention
    Y_pred = model(X, adj, mask)
    assert Y_pred.shape == (batch_size,)

    # Test forward pass with attention
    Y_pred, att = model(X, adj, mask, return_att=True)
    assert Y_pred.shape == (batch_size,)
    assert att.shape == (batch_size, bag_size)



def test_sm_transformer_abmil_compute_loss():
    # Define input shape
    in_dim = 256
    in_shape = (in_dim,)
    # Initialize model
    model = SmTransformerABMIL(in_shape=in_shape)
    # Create dummy input and target
    batch_size = 2
    bag_size = 4
    X = torch.randn(batch_size, bag_size, *in_shape)
    adj = torch.randn(batch_size, bag_size, bag_size)
    mask = torch.randint(0, 2, (batch_size, bag_size)).bool()
    Y = torch.randint(0, 2, (batch_size,)).float()  # Binary labels

    # Test compute_loss
    Y_pred, loss_dict = model.compute_loss(Y, X, adj, mask)
    assert Y_pred.shape == (batch_size,)
    assert isinstance(loss_dict, dict)
    assert "BCEWithLogitsLoss" in loss_dict
    assert loss_dict["BCEWithLogitsLoss"].shape == torch.Size([])  # Scalar loss



def test_sm_transformer_abmil_predict():
    # Define input shape
    in_dim = 256
    in_shape = (in_dim,)
    # Initialize model
    model = SmTransformerABMIL(in_shape=in_shape)
    # Create dummy input
    batch_size = 2
    bag_size = 4
    X = torch.randn(batch_size, bag_size, *in_shape)
    adj = torch.randn(batch_size, bag_size, bag_size)
    mask = torch.randint(0, 2, (batch_size, bag_size)).bool()

    # Test predict without instance predictions
    Y_pred = model.predict(X, adj, mask, return_inst_pred=False)
    assert Y_pred.shape == (batch_size,)

    # Test predict with instance predictions
    Y_pred, y_inst_pred = model.predict(X, adj, mask, return_inst_pred=True)
    assert Y_pred.shape == (batch_size,)
    assert y_inst_pred.shape == (batch_size, bag_size)
