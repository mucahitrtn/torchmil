import torch

class MILModel(torch.nn.Module):
    def __init__(
            self, 
            *args,
            **kwargs
        ):
        super(MILModel, self).__init__()
    
    def forward(self, X, *args, **kwargs):
        """
        Input:
            X: tensor (batch_size, bag_size, ...)
        Output:
            T_pred: tensor (batch_size,)
            ...
        """
        raise NotImplementedError

    def compute_loss(self, T_labels, X, *args, **kwargs):
        """
        Input:
            T_labels: tensor (batch_size,)
            X: tensor (batch_size, bag_size, ...)
        Output:
            loss: tensor (1,)
            T_pred: tensor (batch_size,)
        """
        raise NotImplementedError

    def predict(self, X, *args, return_y_pred=True, **kwargs):
        """
        Input:
            X: tensor (batch_size, bag_size, ...)
        Output:
            T_pred: tensor (batch_size,)
            y_pred: tensor (batch_size, bag_size) if return_y_pred is True
        """
        raise NotImplementedError