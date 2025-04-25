import torch
import pytest
from tensordict import TensorDict
from typing import Any

# Import the modules to be tested
from torchmil.models.mil_model import MILModel, MILModelWrapper, get_args_names  # Replace 'your_module'

# Test get_args_names function
def test_get_args_names():
    def sample_function(self, X, Y, Z=None):
        pass
    args_names = get_args_names(sample_function)
    assert args_names == ('X', 'Y', 'Z')

    def sample_function_no_self(X, Y, Z=None):
        pass
    args_names = get_args_names(sample_function_no_self)
    assert args_names == ('X', 'Y', 'Z')

# Define a dummy MILModel for testing MILModelWrapper
class DummyMILModel(MILModel):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, X, adj=None, **kwargs):
        if adj is not None:
            X = X + adj  # Use adj if provided
        x = self.linear1(X)
        x = torch.relu(x)
        Y_pred = self.linear2(x).mean(dim=(1,2))  # Average over the bag
        return Y_pred

    def compute_loss(self, Y, X, adj=None, **kwargs):
        Y_pred = self.forward(X, adj, **kwargs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Y_pred, Y)
        return Y_pred, {"loss": loss}

    def predict(self, X, adj=None, return_inst_pred=False, **kwargs):
        Y_pred = self.forward(X, adj, **kwargs)
        if return_inst_pred:
            y_inst_pred = torch.sigmoid(Y_pred).unsqueeze(1).expand(-1, X.size(1))  # Expand to bag size
            return Y_pred, y_inst_pred
        return Y_pred

# Test MILModelWrapper
def test_mil_model_wrapper():
    batch_size = 2
    bag_size = 10
    input_dim = 128
    hidden_dim = 64
    output_dim = 1  # Binary classification
    # Create a dummy model instance
    dummy_model = DummyMILModel(input_dim, hidden_dim, output_dim)
    # Wrap it with MILModelWrapper
    wrapped_model = MILModelWrapper(dummy_model)

    # Create a sample TensorDict
    X = torch.randn(batch_size, bag_size, input_dim)
    adj = torch.randn(batch_size, bag_size, input_dim)  # Example adjacency matrix
    Y = torch.randint(0, 2, (batch_size,)).float()
    bag = TensorDict({"X": X, "Y": Y, "adj": adj}, batch_size=[batch_size])

    # Test forward pass
    Y_pred = wrapped_model(bag)
    assert Y_pred.shape == (batch_size,), "Output shape should be (batch_size,)"

    # Test compute_loss
    Y_pred, loss_dict = wrapped_model.compute_loss(bag)
    assert Y_pred.shape == (batch_size,), "Output shape should be (batch_size,)"
    assert isinstance(loss_dict, dict), "Loss should be a dictionary"
    assert "loss" in loss_dict, "Loss dict should contain the loss"

    # Test predict
    Y_pred, y_inst_pred = wrapped_model.predict(bag, return_inst_pred=True)
    assert Y_pred.shape == (batch_size,), "Output shape should be (batch_size,)"
    assert y_inst_pred.shape == (batch_size, bag_size), "Instance prediction shape should be (batch_size, bag_size)"

    Y_pred = wrapped_model.predict(bag, return_inst_pred=False)
    assert Y_pred.shape == (batch_size,), "Output shape should be (batch_size,)"
