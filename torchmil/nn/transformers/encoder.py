import torch

class Encoder(torch.nn.Module):
    """
    Generic encoder class.    
    """

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        add_self: bool = False
    ):
        """
        Class constructor

        Arguments:
            layers: List of encoder layers.
            add_self: Whether to add input to output.
        """
        super(Encoder, self).__init__()
        self.add_self = add_self
        self.layers = layers

    def forward(
        self,
        X: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward method.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.

        Returns:
            Y: Output tensor of shape `(batch_size, bag_size, in_dim)`.        
        """

        Y = X  # (batch_size, bag_size, in_dim)
        for layer in self.layers:
            Y = layer(Y, **kwargs)
            if self.add_self:
                Y = Y + X
        return Y
