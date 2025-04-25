import torch

class Encoder(torch.nn.Module):
    r"""
    Generic Transformer encoder class.

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times D}$
    and (optional) additional arguments, this module computes:

    \begin{align*}
    \mathbf{X}^{0} & = \mathbf{X} \\
    \mathbf{X}^{l} & = \operatorname{Layer}^{l}\left( \mathbf{X}^{l-1}, \ldots \right), \quad l = 1, \ldots, L \\
    \end{align*}

    where $\ldots$ denotes additional arguments.
    The list of layers, $\operatorname{Layer}^{l}$ for $l = 1, \ldots, L$, is given by the `layers` argument, and should be a subclass of [Layer](./#torchmil.nn.transformers.Layer).

    This module outputs $\operatorname{Encoder}(\mathbf{X}) = \mathbf{X}^{L}$ if `add_self=False`,
    and $\operatorname{Encoder}(\mathbf{X}) = \mathbf{X}^{L} + \mathbf{X}$ if `add_self=True`.
    """

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        add_self: bool = False
    ):
        """
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
