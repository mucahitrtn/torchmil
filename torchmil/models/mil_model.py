import torch

def get_args_names(fn):
    args_names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    # remove self from arg_names if exists
    if 'self' in args_names:
        args_names = args_names[1:]
    return args_names

class MILModel(torch.nn.Module):
    r"""
    Base class for Multiple Instance Learning (MIL) models in torchmil.

    Subclasses should implement the following methods:

    - `forward`: Forward pass of the model. Accepts bag features (and optionally other arguments) and returns bag label logits (and optionally other outputs).
    - `compute_loss`: Compute inner losses of the model. Accepts ground truth bag labels, bag features (and optionally other arguments) and returns bag label predictions and a dictionary of pairs (loss_name, loss_value). By default, the model has no inner losses, so this dictionary is empty.
    - `predict`: Predict bag and (optionally) instance labels. Accepts bag features (and optionally other arguments) and returns bag label predictions (and optionally instance label predictions).
        
    """

    def __init__(
            self,
            *args,
            **kwargs
        ):
        """
        Initializes the module.
        """
        super(MILModel, self).__init__()
    
    def forward(
            self, 
            X : torch.Tensor, 
            *args, **kwargs
        ) -> torch.Tensor:
        """
        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
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

        Y_pred = self.forward(X, *args, **kwargs)
        return Y_pred, {}

    def predict(
            self, 
            X: torch.Tensor, 
            return_inst_pred: bool = False,
            *args, **kwargs
        ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def forward(
        self,
        bag,
        **kwargs
    ) -> torch.Tensor:
        arg_names = get_args_names(self.model.forward)
        arg_dict = {k: bag[k] for k in bag.keys() if k in arg_names}
        return self.model(**arg_dict, **kwargs)

    def compute_loss(
        self, 
        bag, 
        **kwargs
    ) -> tuple[torch.Tensor, dict]:
        arg_names = get_args_names(self.model.compute_loss)
        arg_dict = {k: bag[k] for k in bag.keys() if k in arg_names}
        return self.model.compute_loss(**arg_dict, **kwargs)

    def predict(
        self, 
        bag, 
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        arg_names = get_args_names(self.model.predict)
        arg_dict = {k: bag[k] for k in bag.keys() if k in arg_names}
        return self.model.predict(**arg_dict, **kwargs)