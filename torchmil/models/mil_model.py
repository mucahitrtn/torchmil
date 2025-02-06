import torch

class MILModel(torch.nn.Module):
    def __init__(
            self,
            *args,
            **kwargs
        ):
        super(MILModel, self).__init__()
    
    def forward(
            self, 
            X : torch.Tensor, 
            *args, **kwargs
        ):
        """
        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
        """
        raise NotImplementedError

    def compute_loss(
            self, 
            Y: torch.Tensor, 
            X: torch.Tensor, 
            *args, **kwargs
        ) -> tuple[torch.Tensor, dict]:
        """
        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.

        Returns:
            Y_pred: Bag label prediction of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value.
        """
        raise NotImplementedError

    def predict(
            self, 
            X: torch.Tensor, 
            return_inst_pred: bool = False,
            *args, **kwargs
        ) -> torch.Tensor:
        """
        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
        
        Returns:
            Y_pred: Bag label prediction of shape `(batch_size,)`.
            y_inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        raise NotImplementedError