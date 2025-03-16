import torch

from .mil_model import MILModel

from torchmil.nn import PMFTransformerEncoder, SETransformerEncoder
from torchmil.nn.utils import (
    get_feat_dim,
    SinusoidalPositionalEncodingND,
    get_spatial_representation,
    get_sequential_representation,
)


class SETMIL(MILModel):
    r"""
    SETMIL: Spatial Encoding Transformer-Based Multiple Instance Learning for Pathological Image Analysis (SETMIL) model, proposed in the paper [SETMIL: Spatial Encoding Transformer-Based Multiple Instance Learning for Pathological Image Analysis](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_7).

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$, the model optionally applies a feature extractor, $\text{FeatExt}(\cdot)$, to transform the instance features: $\mathbf{X} = \text{FeatExt}(\mathbf{X}) \in \mathbb{R}^{N \times D}$.

    Then, using three tokens-to-token modules (see [T2T](https://arxiv.org/abs/2101.11986) for further information) working in a pyramid arrangement, it modifies the feature map producing a richer representation with multi-scale context information as:

    $$\mathbf{E} = \text{Concat}(\text{T2T}_{k=3}(\mathbf{X}), \text{T2T}_{k=5}(\mathbf{X}), \text{T2T}_{k=7}(\mathbf{X})),$$

    where $k$ is the kernel size of the convolutional layer in the T2T module. Using $\mathbf{E}$, the model applies a spatial encoding transformer (SET) to learn the spatial relationships between the instances in the bag. Using a Multilayer Perceptron (MLP), the SET module takes the form

    $$\text{SETL}( \mathbf{E} ) = \text{MLP}(\text{MultiheadAttention}^{\text{SET}}(\text{LayerNorm}(\mathbf{E}))),$$

    where $\text{MultiheadAttention}^{\text{SET}}$ is the multi-head self-attention mechanism with relative positional encoding (RPE), which uses the euclidean distance between coordinates, to embed the position and context information.

    The output of the SET module is used to predict the bag label $\hat{Y}$ using a linear layer $\phi$:

    $$\hat{Y} = \phi\left( \text{SETL}( \underset{(6)}{\cdots} \text{SETL}(\mathbf{E} )))\right),$$
    """

    def __init__(
        self,
        in_shape: tuple,
        att_dim: int = 512,
        use_pmft: bool = False,
        pmft_n_heads: int = 4,
        pmft_use_mlp: bool = True,
        pmft_dropout: float = 0.0,
        pmft_kernel_list: list[tuple[int, int]] = [(3, 3), (5, 5), (7, 7)],
        pmft_stride_list: list[tuple[int, int]] = [(1, 1), (1, 1), (1, 1)],
        pmft_padding_list: list[tuple[int, int]] = [(1, 1), (2, 2), (3, 3)],
        pmft_dilation_list: list[tuple[int, int]] = [(1, 1), (1, 1), (1, 1)],
        set_n_layers: int = 1,
        set_n_heads: int = 4,
        set_use_mlp: bool = True,
        set_dropout: float = 0.0,
        rpe_ratio: float = 1.9,
        rpe_method: str = "product",
        rpe_mode: str = "ctx",
        rpe_shared_head: bool = True,
        rpe_skip: int = 1,
        rpe_on: str = "k",
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        """
        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension).
            att_dim: Attention dimension.
            use_pmft: If True, use Pyramid Multihead Feature Transformer (PMFT) before the SET module.
            pmft_n_heads: Number of heads in the PMFT module.
            pmft_use_mlp: If True, use MLP in the PMFT module.
            pmft_dropout: Dropout rate in the PMFT module.
            pmft_kernel_list: List of kernel sizes in the PMFT module.
            pmft_stride_list: List of stride sizes in the PMFT module.
            pmft_padding_list: List of padding sizes in the PMFT module.
            pmft_dilation_list: List of dilation sizes in the PMFT module.
            set_n_layers: Number of layers in the SET module.
            set_n_heads: Number of heads in the SET module.
            set_use_mlp: If True, use MLP in the SET module.
            set_dropout: Dropout rate in the SET module.
            rpe_ratio: Ratio for relative positional encoding.
            rpe_method: Method for relative positional encoding. Possible values: 'product', 'concat'.
            rpe_mode: Mode for relative positional encoding. Possible values: 'ctx', 'ctx+pos'.
            rpe_shared_head: If True, share the attention head in the relative positional encoding.
            rpe_skip: Number of layers to skip for relative positional encoding.
            rpe_on: Apply relative positional encoding on 'q', 'k', or 'v'.
            feat_ext: Feature extractor.
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits.
        """
        super().__init__()
        self.criterion = criterion

        self.feat_ext = feat_ext
        feat_dim = get_feat_dim(feat_ext, in_shape)

        if feat_dim != att_dim:
            self.proj = torch.nn.Linear(feat_dim, att_dim)
        else:
            self.proj = torch.nn.Identity()
        self.att_dim = att_dim

        self.pos_embed = SinusoidalPositionalEncodingND(1, att_dim)

        self.use_pmft = use_pmft
        if use_pmft:
            self.pmf_transf = PMFTransformerEncoder(
                in_dim=att_dim,
                att_dim=att_dim,
                out_dim=att_dim,
                kernel_list=pmft_kernel_list,
                stride_list=pmft_stride_list,
                padding_list=pmft_padding_list,
                dilation_list=pmft_dilation_list,
                n_heads=pmft_n_heads,
                use_mlp=pmft_use_mlp,
                dropout=pmft_dropout,
            )

        # TODO: In the paper they mention that they use the LN BEFORE the transformer, but in the code it is AFTER the transformer.
        self.se_transf = SETransformerEncoder(
            in_dim=att_dim,
            att_dim=att_dim,
            n_layers=set_n_layers,
            n_heads=set_n_heads,
            use_mlp=set_use_mlp,
            dropout=set_dropout,
            rpe_ratio=rpe_ratio,
            rpe_method=rpe_method,
            rpe_mode=rpe_mode,
            rpe_shared_head=rpe_shared_head,
            rpe_skip=rpe_skip,
            rpe_on=rpe_on,
        )

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, att_dim))

        self.classifier = torch.nn.Linear(in_features=att_dim, out_features=1)

    def _pad_to_square(self, X: torch.Tensor) -> torch.Tensor:
        """
        Pad tensor to square shape.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, coord1, coord2)`.

        Returns:
            X: Padded tensor of shape `(batch_size, bag_size, max_dim, max_dim)`.
        """
        max_dim = max(X.size(-2), X.size(-1))
        pad_h = max_dim - X.size(-2)
        pad_w = max_dim - X.size(-1)
        X = torch.nn.functional.pad(X, (0, pad_w, 0, pad_h))
        return X

    def forward(
        self, X: torch.Tensor, coords: torch.Tensor, return_att: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            coords: Coordinates of shape `(batch_size, bag_size, coord_dim)`.
            return_att: If True, returns attention values (before normalization) in addition to `Y_pred`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
        """

        assert coords.shape[2] == 2, "SETMIL only supports 2D coordinates."

        batch_size = X.size(0)
        orig_bag_size = X.size(1)

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)
        X = self.proj(X)  # (batch_size, bag_size, att_dim)

        X = get_spatial_representation(
            X, coords
        )  # (batch_size, coord1, coord2, att_dim)

        # TODO: Does this transposition work as intended? Isnt the final shape (batch_size, att_dim, coord_2, coord_1)?
        X = X.transpose(1, -1)  # (batch_size, att_dim, coord1, coord2)
        X = self._pad_to_square(X)  # (batch_size, att_dim, max_coord, max_coord)

        coords_shape = X.size()[2:]

        if self.use_pmft:
            X = self.pmf_transf(X)  # (batch_size, seq_len, att_dim)
        else:
            X = X.reshape(batch_size, self.att_dim, -1).transpose(
                1, 2
            )  # (batch_size, seq_len, att_dim)
            # seq_len = coord1 * coord2

        # Add class token
        cls_token = self.cls_token.expand(
            batch_size, -1, -1
        )  # (batch_size, 1, att_dim)
        X = torch.cat([cls_token, X], dim=1)  # (batch_size, seq_len + 1, att_dim)

        pos = self.pos_embed(X)  # (batch_size, seq_len + 1, att_dim)
        X = X + pos  # (batch_size, seq_len + 1, att_dim)

        if return_att:
            if self.use_pmft:
                print(
                    "Warning: bag size has changed after PMF Transformer, cannot return attention values."
                )
                att = torch.zeros(batch_size, orig_bag_size, device=X.device)
            else:
                X, att = self.se_transf(
                    X, return_att=True
                )  # (batch_size, seq_len + 1, att_dim), (batch_size, seq_len+1, bag_size+1)
                att = att[:, 0, 1:]  # (batch_size, seq_len)
                att = att.view(
                    batch_size, *coords_shape
                )  # (batch_size, coord1, coord2)
                att = att.unsqueeze(-1)  # (batch_size, coord1, coord2, 1)
                att = get_sequential_representation(
                    att, coords
                )  # (batch_size, bag_size, 1)
                att = att.squeeze(-1)  # (batch_size, bag_size)
        else:
            X = self.se_transf(X)  # (batch_size, bag_size' + 1, att_dim)

        z = X[:, 0]  # (batch_size, att_dim)

        Y_pred = self.classifier(z)  # (batch_size, 1)

        if return_att:
            return Y_pred, att
        else:
            return Y_pred

    def compute_loss(
        self,
        Y: torch.Tensor,
        X: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            coords: Coordinates of shape `(batch_size, bag_size, coord_dim)`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value.
        """
        Y_pred = self.forward(X, coords, return_att=False)

        crit_loss = self.criterion(Y_pred.float(), Y.float())
        crit_name = self.criterion.__class__.__name__

        return Y_pred, {crit_name: crit_loss}

    def predict(
        self, X: torch.Tensor, coords: torch.Tensor, return_inst_pred: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            coords: Coordinates of shape `(batch_size, bag_size, coord_dim)`.
            return_inst_pred: If `True`, returns instance labels predictions, in addition to bag label predictions.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            y_inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        return self.forward(X, coords, return_att=return_inst_pred)
