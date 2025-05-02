import torch

from torchmil.nn import (
    MultiheadSelfAttention,
    MultiheadCrossAttention,
    TransformerEncoder,
)
from torchmil.nn.utils import get_feat_dim


class IIBMILDecoderLayer(torch.nn.Module):
    """
    IIB-MIL decoder layer as described in the paper [IIB-MIL: Integrated Instance-Level and Bag-Level Multiple Instances Learning with Label Disambiguation for Pathological Image Analysis](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_54).
    """

    def __init__(
            self, 
            dim: int = 256, 
            n_heads: int = 4, 
            dropout: float = 0.0,
            use_mlp: bool = True
    ): 
        """
        Arguments:
            dim: Embedding dimension.
            n_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()

        self.cross_att = MultiheadCrossAttention(dim, dim, n_heads)
        self.norm1 = torch.nn.LayerNorm(dim)

        self.self_att = MultiheadSelfAttention(dim, dim, n_heads)
        self.norm2 = torch.nn.LayerNorm(dim)

        if use_mlp:
            self.mlp_layer = torch.nn.Sequential(
                torch.nn.Linear(dim, 4 * dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(4 * dim, dim),
                torch.nn.Dropout(dropout),
            )
        else:
            self.mlp_layer = torch.nn.Identity()

    def forward(
        self,
        U: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Arguments:
            U: Input embeddings of shape `(batch_size, n_queries, dim)`.
            X: Input embeddings of shape `(batch_size, n_instances, dim)`.
            mask: Mask tensor of shape `(batch_size, n_instances)`.

        Returns:
            Z: Output embeddings of shape `(batch_size, n_queries, dim)`.
        """

        X = X + self.self_att(X, mask)  # (batch_size, n_instances, dim)
        X = self.norm1(X)  # (batch_size, n_instances, dim)

        U = U + self.cross_att(U, X, mask)  # (batch_size, n_queries, dim)
        U = self.norm2(U)  # (batch_size, n_queries, dim)

        U = self.mlp_layer(U)  # (batch_size, n_queries, dim)

        return U


class IIBMILDecoder(torch.nn.Module):
    """
    IIB-MIL decoder as described in the paper [IIB-MIL: Integrated Instance-Level and Bag-Level Multiple Instances Learning with Label Disambiguation for Pathological Image Analysis](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_54).
    """

    def __init__(
        self,
        dim: int = 256,
        n_layers: int = 1,
        n_heads: int = 4,
        use_mlp: bool = True,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [IIBMILDecoderLayer(dim, n_heads, use_mlp) for _ in range(n_layers)]
        )

    def forward(
        self,
        U: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Arguments:
            U: Input embeddings of shape `(batch_size, n_queries, dim)`.
            X: Input embeddings of shape `(batch_size, n_instances, dim)`.
            mask: Mask tensor of shape `(batch_size, n_instances)`.

        Returns:
            Z: Output embeddings of shape `(batch_size, n_queries, dim)`.
        """
        for layer in self.layers:
            U = layer(U, X, mask)
        return X


class IIBMIL(torch.nn.Module):
    r"""
    Integrated Instance-Level and Bag-Level Multiple Instance Learning (IIB-MIL) model, proposed in the paper [IIB-MIL: Integrated Instance-Level and Bag-Level Multiple Instances Learning with Label Disambiguation for Pathological Image Analysis](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_54).

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$, the model optionally applies a feature extractor, $\text{FeatExt}(\cdot)$, to transform the instance features: $\mathbf{X} = \text{FeatExt}(\mathbf{X}) \in \mathbb{R}^{N \times D}$.

    Then, a [TransformerEncoder](../nn/transformers/conventional_transformer.md) is applied to transform the instance features using context information.
    Subsequently, the model uses **bag-level** and **instance-level** supervision:

    **Bag-level supervision**: The instances are aggregated into a class token using $\texttt{n_queries}$ queries embeddings and the [IIBMILDecoder](./#torchmil.models.iibmil.IIBMILDecoder). A linear layer is then applied to predict the bag label.

    **Instance-level supervision**: Consists of four steps.

    1. Using an instance classifier, obtain the probability of instance $i$ belonging to class $c$, denoted as $p_{i,c}$.
    2. The prototype $\mathbf{p}_{c,t} \in \mathbf{R}^{D}$ of class $c$ at time $t$ is updated using a momentum update rule based on the set of instances with the top $k$ highest probabilities of belonging to class $c$. Writing $\mathbf{P}_t = \left[ \mathbf{p}_{1,t}, \ldots, \mathbf{p}_{C,t}  \right]^\top \in \mathbb{R}^{C \times D}$, the prototype label $z_{i}$ of each instance is obtained as $z_{i} = \text{argmax}_{c} \ \mathbf{P} \mathbf{x}_i$.
    3. Compute instance-level soft labels using the prototype labels and a momentum update.
    4. Compute the instance-level cross-entropy loss using the soft labels and the instance classifier.
    """

    def __init__(
        self,
        in_shape: tuple = None,
        att_dim: int = 256,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        use_mlp_encoder: bool = True,
        use_mlp_decoder: bool = True,
        n_heads: int = 4,
        n_queries: int = 5,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        criterion : torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        """
        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension). If not provided, it will be lazily initialized.
            att_dim: Attention dimension.
            n_layers_encoder: Number of layers in the transformer encoder.
            n_layers_decoder: Number of layers in the transformer decoder.
            use_mlp_encoder: If True, uses a multi-layer perceptron (MLP) in the encoder.
            use_mlp_decoder: If True, uses a multi-layer perceptron (MLP) in the decoder.
            n_heads: Number of attention heads.
            n_queries: Number of queries.
            feat_ext: Feature extractor.
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits.
        """
        super().__init__()
        self.criterion = criterion
        self.feat_ext = feat_ext

        feat_dim = get_feat_dim(feat_ext, in_shape)

        self.query_embed = torch.nn.Embedding(n_queries, att_dim)

        if feat_dim != att_dim:
            self.feat_proj = torch.nn.Linear(feat_dim, att_dim)
        else:
            self.feat_proj = torch.nn.Identity()

        self.encoder = TransformerEncoder(
            att_dim, att_dim, n_heads, n_layers_encoder, use_mlp=use_mlp_encoder, add_self=False
        )
        self.decoder = IIBMILDecoder(att_dim, n_layers_decoder, n_heads, use_mlp_decoder)

        self.inst_classifier = torch.nn.Linear(att_dim, 1)
        self.bag_classifier = torch.nn.Linear(n_queries * att_dim, 1)

        self.register_buffer("prototypes", torch.zeros(2, att_dim))

        self.criterion = criterion

    def _inst_loss(
        self,
        X_enc: torch.Tensor,
        y_pred: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute instance-level loss.

        Arguments:
            X_enc: Instance embeddings of shape `(batch_size, bag_size, att_dim)`.
            y_pred: Instance label logits of shape `(batch_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`.

        Returns:
            loss_instance: Instance-level loss.
        """
        batch_size, bag_size, _ = X_enc.shape

        if self.training:
            self._update_prototypes(y_pred, X_enc, mask)

        # Compute protoypical logits
        X_enc = X_enc.view(batch_size * bag_size, -1)  # (batch_size * bag_size, dim)
        y_pred = y_pred.view(batch_size * bag_size)  # (batch_size * bag_size)
        prototypes = self.prototypes.clone().detach()  # (2, dim)
        logits_prot = torch.mm(X_enc.detach(), prototypes.t()).squeeze(1)  # (batch_size * bag_size, 2)
        score_prot = torch.softmax(logits_prot, dim=1)  # (batch_size * bag_size, 2)

        # compute pseudo labels
        pseudo_labels = torch.argmax(score_prot, dim=1)  # (batch_size * bag_size)
        pseudo_labels = pseudo_labels.type(torch.float32)  # (batch_size * bag_size)

        # compute instance loss
        loss_instance = self.criterion(y_pred, pseudo_labels)  # (1,)

        return loss_instance

    def _update_prototypes(
        self,
        y_pred: torch.Tensor,
        X_enc: torch.Tensor,
        mask: torch.Tensor = None,
        proto_m: float = 0.99,
    ) -> None:
        """
        Update prototypes.

        Arguments:
            y_pred: Instance label logits of shape `(batch_size, bag_size)`.
            X_enc: Instance embeddings of shape `(batch_size, bag_size, att_dim)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            proto_m: Momentum for updating prototypes

        Returns:
            None
        """
        batch_size, bag_size, feat_dim = X_enc.shape

        X_enc = X_enc.view(batch_size * bag_size, -1)  # (batch_size * bag_size, dim)
        y_pred = y_pred.view(batch_size * bag_size)  # (batch_size * bag_size)

        if mask is not None:
            mask = mask.view(batch_size * bag_size)  # (batch_size * bag_size)
            mask = mask.bool()

            y_pred = torch.masked_select(
                y_pred,
                mask,
            )  # (batch_size * bag_size,)

            X_enc = torch.masked_select(
                X_enc,
                mask.unsqueeze(-1).repeat(1, feat_dim),
            ).reshape(
                -1, feat_dim
            )  # (batch_size * bag_size, dim)

        k = y_pred.shape[0] // 10
        _, indice_0 = torch.topk(
            y_pred,
            k,
            dim=-1,
            largest=True,
            sorted=True,
            out=None,
        )  # (batch_size * bag_size, k)

        _, indice_1 = torch.topk(
            y_pred,
            k,
            dim=-1,
            largest=False,
            sorted=True,
            out=None,
        )  # (batch_size * bag_size, k)

        X_enc_0 = X_enc[indice_0, :]  # (batch_size * bag_size, k, dim)
        X_enc_1 = X_enc[indice_1, :]  # (batch_size * bag_size, k, dim)

        for i in range(len(indice_0)):
            self.prototypes[0, :] = (
                self.prototypes[0, :] * proto_m + (1 - proto_m) * X_enc_0[i]
            )

        for i in range(len(indice_1)):
            self.prototypes[1, :] = (
                self.prototypes[1, :] * proto_m + (1 - proto_m) * X_enc_1[i]
            )

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor = None,
        return_inst_pred: bool = False,
        return_X_enc: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_inst_pred: If True, returns attention values (before normalization) in addition to `Y_pred`.
            return_X_enc: If True, returns instance embeddings in addition to `Y_pred`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            y_inst_pred: Only returned when `return_inst_pred=True`. Instance label logits of shape `(batch_size, bag_size)`.
            X_enc: Only returned when `return_X_enc=True`. Instance embeddings of shape `(batch_size, bag_size, att_dim)`.
        """

        batch_size = X.shape[0]

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)
        X = self.feat_proj(X)  # (batch_size, bag_size, att_dim)
        X_enc = self.encoder(X, mask)  # (batch_size, bag_size, att_dim)

        y_pred = self.inst_classifier(X_enc).squeeze(-1)  # (batch_size, bag_size)

        U = self.query_embed.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (batch_size, n_queries, att_dim)

        z = self.decoder(X_enc, U)  # (batch_size, n_queries, att_dim)
        z = z.view(batch_size, -1)  # (batch_size, n_queries * att_dim)

        Y_pred = self.bag_classifier(z).squeeze(-1)  # (batch_size,)

        if return_inst_pred:
            if return_X_enc:
                return Y_pred, y_pred, X_enc
            else:
                return Y_pred, y_pred
        else:
            if return_X_enc:
                return Y_pred, X_enc
            else:
                return Y_pred

    def compute_loss(
        self,
        Y: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value.
        """
        Y_pred, y_pred, X_enc = self.forward(
            X, mask, return_inst_pred=True, return_X_enc=True
        )
        inst_loss = self._inst_loss(X_enc, y_pred, mask)
        crit_loss = self.criterion(Y_pred.float(), Y.float())
        crit_name = self.criterion.__class__.__name__
        return Y_pred, {crit_name: crit_loss, "InstLoss": inst_loss}

    def predict(
        self,
        X: torch.Tensor,
        mask: torch.Tensor = None,
        return_inst_pred: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_inst_pred: If `True`, returns instance labels predictions, in addition to bag label predictions.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            y_inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        return self.forward(X, mask, return_inst_pred=return_inst_pred)
