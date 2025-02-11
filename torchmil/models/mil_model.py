import torch

def get_args_names(fn):
    args_names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    # remove self from arg_names if exists
    if 'self' in args_names:
        args_names = args_names[1:]
    return args_names

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

class MILModelWrapper(MILModel):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, bag, **kwargs):
        arg_names = get_args_names(self.model.forward)
        arg_dict = {k: bag[k] for k in bag.keys() if k in arg_names}
        return self.model(**arg_dict, **kwargs)

    def compute_loss(self, bag, **kwargs):
        arg_names = get_args_names(self.model.compute_loss)
        arg_dict = {k: bag[k] for k in bag.keys() if k in arg_names}
        return self.model.compute_loss(**arg_dict, **kwargs)

    def predict(self, bag, **kwargs):
        arg_names = get_args_names(self.model.predict)
        arg_dict = {k: bag[k] for k in bag.keys() if k in arg_names}
        return self.model.predict(**arg_dict, **kwargs)