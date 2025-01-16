import copy

import torch
import torch.nn.functional as F
from torch import nn


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        p_dropout=0.1,
        activation="relu",
        n_heads=8,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=p_dropout)
        self.dropout1 = nn.Dropout(p_dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=p_dropout)
        self.dropout2 = nn.Dropout(p_dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(p_dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(p_dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, src):
        q = k = v = tgt

        tgt2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1)
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        q = tgt
        k = v = src

        tgt2 = self.cross_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))[
            0
        ].transpose(0, 1)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, src):
        output = tgt
        for _, layer in enumerate(self.layers):
            output = layer(output, src)

        return output


# Encoder
class IIBMIL_Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        dim,
        depth=1,
        p_dropout=0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.patch_to_embedding = nn.Linear(input_dim, dim)
        self.z_dim = dim
        self.dropout = nn.Dropout(p_dropout)
        self.to_cls_token = nn.Identity()

        self.deep = {}
        self.res = {}
        self.depth = depth
        for i in range(self.depth):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = self.z_dim

            self._modules["depth_{}".format(i)] = nn.Sequential(
                nn.Linear(in_dim, self.z_dim * 2),
                nn.Tanh(),
                nn.LayerNorm(self.z_dim * 2),
                nn.Dropout(p_dropout),
                nn.Linear(self.z_dim * 2, self.z_dim * 4),
                nn.Tanh(),
                nn.LayerNorm(self.z_dim * 4),
                nn.Dropout(p_dropout),
                nn.Linear(self.z_dim * 4, self.z_dim * 2),
                nn.Tanh(),
                nn.LayerNorm(self.z_dim * 2),
                nn.Dropout(p_dropout),
                nn.Linear(self.z_dim * 2, self.z_dim * 2),
                nn.Tanh(),
                nn.LayerNorm(self.z_dim * 2),
                nn.Dropout(p_dropout),
                nn.Linear(self.z_dim * 2, self.z_dim * 1),
                nn.Tanh(),
                nn.LayerNorm(self.z_dim * 1),
                nn.Dropout(p_dropout)
            )
            self._modules["encode_{}".format(i)] = nn.TransformerEncoderLayer(
                d_model=self.z_dim, nhead=2
            )
            self._modules["trans_{}".format(i)] = nn.TransformerEncoder(
                self._modules["encode_{}".format(i)], num_layers=1
            )
            self._modules["res_{}".format(i)] = nn.Linear(input_dim, self.z_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.z_dim * 1), nn.Linear(self.z_dim * 1, 1)
        )

    def forward(self, X, adj_mat, mask):

        x = X
        b, n, dimen = X.shape

        for i in range(self.depth):
            x = self._modules["depth_{}".format(i)](x)
            x = self._modules["trans_{}".format(i)](x.transpose(0, 1))
            x = x.transpose(0, 1)
            x = x + torch.relu(self._modules["res_{}".format(i)](X))

        return (self.mlp_head(x), x)


# Decoder
class IIBMIL_Decoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_decoder_layers=2,
        dim_feedforward=1024,
        p_dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(
            d_model,
            dim_feedforward,
            p_dropout,
            activation,
            nhead
        )
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

    def forward(self, tgt, memory):
        hs = self.decoder(tgt, memory)
        return hs


# partial loss + bag loss
class IIBMIL(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim=128,
        depth_encoder=1,
        depth_decoder=1,
        num_queries=5, 
        criterion=torch.nn.BCEWithLogitsLoss(),
    ):
        super().__init__()
        self.patch_encoder = IIBMIL_Encoder(input_dim, emb_dim, depth_encoder)

        self.wsi_aggregator = IIBMIL_Decoder(
            d_model=emb_dim,
            nhead=4,
            num_decoder_layers=depth_decoder,
            dim_feedforward=emb_dim * 2,
            p_dropout=0.1,
            activation="relu",
        )
        self.query_embed = nn.Embedding(num_queries, emb_dim)

        self.wsi_classifier = nn.Linear(emb_dim * num_queries, 1)

        self.register_buffer("prototypes", torch.zeros(2, emb_dim))

        self.criterion = criterion

        self.num_prot_updates = 0
        self.max_prot_updates = 1000

    def forward(self, X, adj_mat, mask, *args, return_y_pred=False, **kwargs):
        bs, patch_num, _ = X.shape

        patch_classifier_output, x_instance = self.patch_encoder(X, adj_mat, mask) # (bs, patch_num, 1), (bs, patch_num, dim)

        x_instance = x_instance.reshape(bs, patch_num, -1) # (bs, patch_num, dim)

        tgt = self.query_embed.weight
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        hs = self.wsi_aggregator(tgt, x_instance)
        wsi_classifier_output = self.wsi_classifier(hs.view(hs.shape[0], -1)).squeeze(1)

        # if return_inst_loss:
        #     if return_y_pred:
        #         return wsi_classifier_output, patch_classifier_output, loss_instance
        #     else:
        #         return wsi_classifier_output, loss_instance
        # else:
        #     if return_y_pred:
        #         return wsi_classifier_output, patch_classifier_output
        #     else:
        #         return wsi_classifier_output

        if return_y_pred:
            patch_classifier_output = patch_classifier_output.reshape(bs, patch_num)
            return wsi_classifier_output, patch_classifier_output
        else:
            return wsi_classifier_output

        # return (patch_classifier_output, x_instance, score_prot, wsi_classifier_output)

    def compute_inst_loss(self, X, adj_mat, mask, *args, **kwargs):
        bs, patch_num, patch_dim = X.shape

        patch_classifier_output, x_instance = self.patch_encoder(X, adj_mat, mask) # (bs, patch_num, 1), (bs, patch_num, dim)
        x_instance = x_instance.reshape(bs * patch_num, -1) # (bs * patch_num, dim)
        patch_classifier_output = patch_classifier_output.reshape(bs * patch_num) # (bs * patch_num)

        # compute protoypical logits
        self.update_prototypes(patch_classifier_output, x_instance, mask)
        prototypes = self.prototypes.clone().detach() # (2, dim)
        logits_prot = torch.mm(x_instance.detach(), prototypes.t()).squeeze(1) # (bs * patch_num, 2)
        score_prot = torch.softmax(logits_prot, dim=1) # (bs * patch_num, 2)
        pseudo_labels = torch.argmax(score_prot, dim=1) # (bs * patch_num)
        pseudo_labels = pseudo_labels.type(torch.float32)
        
        # compute instance loss
        loss_instance = self.criterion(patch_classifier_output, pseudo_labels)
        # loss_instance = torch.mean(loss_instance, dim=1)

        return loss_instance
    
    def update_prototypes(
        self, patch_cls, x_instance, mask, proto_m=0.99
    ):
        bs = mask.shape[0]
        num_patches = mask.shape[1]
        feat_dim = x_instance.shape[1]
        mask = mask.reshape(bs*num_patches)
        mask = mask.bool()
        # patch_cls = patch_cls[wsi_idx.squeeze(-1), :, :]
        # x_instance = x_instance[wsi_idx.squeeze(-1), :, :]
        # mask = mask.reshape(-1, N)[wsi_idx.squeeze(-1), :]

        patch_cls_select = torch.masked_select(
            patch_cls,
            mask,
        )
        x_instance_select = torch.masked_select(
            x_instance.reshape(bs * num_patches, -1),
            mask.unsqueeze(-1).repeat(1, 1, feat_dim).reshape(bs * num_patches, -1),
        ).reshape(-1, feat_dim)
        
        # predicted_scores_select = torch.softmax(patch_cls_select, dim=0)

        topk = patch_cls_select.shape[0] // 10
        _, indice_0 = torch.topk(
            patch_cls_select,
            topk,
            dim=-1,
            largest=True,
            sorted=True,
            out=None,
        )

        _, indice_1 = torch.topk(
            patch_cls_select,
            topk,
            dim=-1,
            largest=False,
            sorted=True,
            out=None,
        )

        # proto_m = 1 - (1 - proto_m) * self.num_prot_updates / self.max_prot_updates
        proto_m = proto_m * (1.0 - self.num_prot_updates / self.max_prot_updates)

        x_instance_select_0 = x_instance_select[indice_0, :]
        x_instance_select_1 = x_instance_select[indice_1, :]

        for i in range(len(indice_0)):
            self.prototypes[0,:] = self.prototypes[0, :] * proto_m + (1 - proto_m) * x_instance_select_0[i]
        
        for i in range(len(indice_1)):
            self.prototypes[1,:] = self.prototypes[1, :] * proto_m + (1 - proto_m) * x_instance_select_1[i]
            
    def compute_loss(self, T_labels, X, adj_mat, mask, *args, **kwargs):
        """
        Input:
            T_labels: tensor (batch_size,)
            X: tensor (batch_size, bag_size, ...)
            adj_mat: sparse coo tensor (batch_size, bag_size, bag_size)
            mask: tensor (batch_size, bag_size)
        Output:
            T_logits_pred: tensor (batch_size,)
            loss_dict: dict {'BCEWithLogitsLoss', **pool_loss_dict, ...}
        """
        T_logits_pred = self.forward(X, adj_mat, mask)
        inst_loss = self.compute_inst_loss(X, adj_mat, mask)
        crit_loss = self.criterion(T_logits_pred.float(), T_labels.float())
        crit_name = self.criterion.__class__.__name__
        return T_logits_pred, { crit_name : crit_loss, 'InstLoss' : inst_loss }
    
    @torch.no_grad()
    def predict(self, X, adj_mat, mask, *args, return_y_pred=True, **kwargs):
        T_logits_pred, y_pred = self.forward(X, adj_mat, mask, return_y_pred=return_y_pred)
        return T_logits_pred, y_pred
