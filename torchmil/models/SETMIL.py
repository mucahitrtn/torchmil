import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
from .modules.t2t_module.token_transformer import Token_transformer
from .modules.t2t_module.transformer_block import Block, get_sinusoid_encoding



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'T2t_vit_7': _cfg(),
    'T2t_vit_10': _cfg(),
    'T2t_vit_12': _cfg(),
    'T2t_vit_14': _cfg(),
    'T2t_vit_19': _cfg(),
    'T2t_vit_24': _cfg(),
    'T2t_vit_t_14': _cfg(),
    'T2t_vit_t_19': _cfg(),
    'T2t_vit_t_24': _cfg(),
    'T2t_vit_14_resnext': _cfg(),
    'T2t_vit_14_wide': _cfg(),
}

class ASPPBlock(nn.Module):
    def __init__(self, in_chans, embed_dim=768, K=3, S=1, P=2, D=1, **kwargs):
        super().__init__()

        self.soft_split = nn.Unfold(kernel_size=(K, K), stride=(S, S), padding=(P, P), dilation=(D, D))
        self.attention = Token_transformer(dim=in_chans * K * K, in_dim=embed_dim, num_heads=1, mlp_ratio=1.0, **kwargs)

    def forward(self, x):
        # step0: soft split
        x = self.soft_split(x).transpose(1, 2)
        # [bs, stride_times1, kernel_size*kernel_size*in_chans]

        # iteration1: re-structurization/reconstruction
        x = self.attention(x)
        # [B, stride_times1 (new_HW), embed_dim]

        return x


class ASPP(nn.Module):
    def __init__(self, in_chans, embed_dim=768, **kwargs):
        super().__init__()
        ##self.p0 = ASPPBlock(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, K=1,)
        self.p1 = ASPPBlock(in_chans=in_chans, embed_dim=embed_dim, K=3, P=(3-1)//2, **kwargs)
        self.p2 = ASPPBlock(in_chans=in_chans, embed_dim=embed_dim, K=5, P=(5-1)//2, **kwargs)
        self.p3 = ASPPBlock(in_chans=in_chans, embed_dim=embed_dim, K=7, P=(7-1)//2,**kwargs)

    def forward(self, x):
        #x0 = self.p0(x)
        x1 = self.p1(x)
        x2 = self.p2(x)
        x3 = self.p3(x)
        x = torch.cat(( x1, x2, x3), dim=2)
        return x

class Downsample(nn.Module):
    """
    dimensionality reduction
    """
    def __init__(self, in_chans, embed_dim=768):
        super().__init__()
        # self.reduce = ASPPBlock(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, K=k, S=(k-1)//2, P=(k-1)//2)
        self.reduce = nn.Conv1d(in_chans, embed_dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        x = self.reduce(x)
        x = x.view(B, -1, H, W)
        
        # new_HW = int(math.ceil(np.sqrt(HW)))**2
        # if new_HW != HW:
        #     # pad x to have shape (B, new_HW, C)
        #     zeros = torch.zeros(B, new_HW-HW, C).to(x.device)
        #     x = torch.cat([x, zeros], dim=1) # (B, new_HW, C)            
        # x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))

        # B, C, H, W = x.shape
        # x = x.reshape(B, C, -1)
        # x = self.reduce(x)
        # B, C, new_HW = x.shape
        # x = x.reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))

        return x

# @torch.compile()
def _fill_abs_pos(X, Y, coords, bag_indices):
    """
    Input:
        X: tensor (batch_size, H, W, num_feat)
        Y: tensor (batch_size, bag_size, num_feat).
        coords: tensor (num_nonzero, 3)
        bag_indices: tensor (num_nonzero,)
    """

    # num_nonzero = coords.shape[0]
    # for i in range(num_nonzero):
    #     b, h, w = coords[i]
    #     bag_idx = bag_indices[i]
    #     X[b, h, w] = Y[b, bag_idx]

    # Extract indices from coords tensor
    b_indices = coords[:, 0].long()
    h_indices = coords[:, 1].long()
    w_indices = coords[:, 2].long()


    # Use advanced indexing to assign values from Y to X
    X[b_indices, h_indices, w_indices] = Y[b_indices, bag_indices]

    return X

# @torch.compile()
def _fill_bag_seq(X, Y, coords, bag_indices):
    """
    Input:
        X: tensor (batch_size, H, W, num_feat)
        Y: tensor (batch_size, bag_size, num_feat).
        coords: tensor (num_nonzero, 3)
        bag_indices: tensor (num_nonzero,)
    """

    # num_nonzero = coords.shape[0]
    # for i in range(num_nonzero):
    #     b, h, w = coords[i]
    #     bag_idx = bag_indices[i]
    #     Y[b, bag_idx] = X[b, h, w]

    # Extract indices from coords tensor
    b_indices = coords[:, 0].long()
    h_indices = coords[:, 1].long()
    w_indices = coords[:, 2].long()

    # Use advanced indexing to assign values from X to Y
    # This updates values in Y based on positions specified by (b, bag_idx) pairs
    Y[b_indices, bag_indices] = X[b_indices, h_indices, w_indices]

    return Y

class SETMIL(nn.Module):
    def __init__(
        self, 
        in_chans=3, 
        num_classes=1000, 
        embed_dim=768, 
        depth=12,
        num_heads=12, 
        mlp_ratio=4., 
        qkv_bias=False, 
        qk_scale=None, 
        drop_rate=0., 
        attn_drop_rate=0.,
        drop_path_rate=0., 
        norm_layer=nn.LayerNorm, 
        token_dim=64, 
        irpe={
            "rpe": 0,
            "method": 'euc',
            "mode": 'bias',
            "shared_head": True,
            "rpe_on": 'q',
        }, 
        token_t={
            "rpe": 0,
            "drop": 0.1,
            "attn_drop": 0.1,
            "method": 'euc',
            "mode": 'bias',
            "shared_head": True,
            "rpe_on": 'q',
        }, 
        aspp_flag=False, 
        criterion = nn.BCEWithLogitsLoss(reduction='mean'),
        max_token_num = 9000,
        use_channel_reduce=True
        ):
        super().__init__()
        self.aspp_flag = aspp_flag
        self.num_classes = num_classes
        self.embed_dim = embed_dim 
        self.token_dim = token_dim
        self.criterion = criterion
        self.use_channel_reduce = use_channel_reduce
        self.max_token_num = max_token_num

        if self.use_channel_reduce:
            self.channel_reduce = Downsample(in_chans=in_chans, embed_dim=token_dim)
        else:
            self.token_dim = in_chans

        if self.aspp_flag:
            self.aspp = ASPP(in_chans=token_dim, embed_dim=self.embed_dim, **token_t)
            # num_patches = self.aspp.num_patches
            decoder_embed_dim = embed_dim * 3
        else:
            # num_patches = (img_size//(channel_reduce_rate//2))**2
            decoder_embed_dim = token_dim
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=decoder_embed_dim), requires_grad=False)
        # self.pos_embed = get_sinusoid_encoding(n_position=num_patches, d_hid=decoder_embed_dim).detach().numpy()
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, **irpe)
            for i in range(depth)])
        self.norm = norm_layer(decoder_embed_dim)

        # Classifier head
        self.head = nn.Linear(decoder_embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, y, adj_mat, *args, return_att=False, **kwargs):
        """
        Input:
            y: tensor (batch_size, bag_size, num_feat)        
        """

        if not adj_mat.is_sparse:
            raise ValueError("adj_mat should be a sparse tensor")
        
        batch_size = y.shape[0]
        bag_size = y.shape[1]
        num_feat = y.shape[2]
        device = y.device
                
        coords = adj_mat.indices().transpose(0, 1) # (num_nonzero, num_dim)
        bag_indices = adj_mat.values().type(torch.int64) # (num_nonzero,)
        
        x = adj_mat.to_dense().type(torch.float32) # (batch_size, bag_size) or (batch_size, H, W)
        
        if len(x.shape) == 2: # if data is 1D
            H = bag_size
            W = 1
            x = x.unsqueeze(2) # (batch_size, H, 1, num_feat)
            zeros = torch.zeros(coords.shape[0], 1).type(torch.int64).to(device)
            coords = torch.cat([coords, zeros], dim=1)
        else:
            H = x.shape[1]
            W = x.shape[2]
        # At this point, x has shape [batch_size, H, W], coords has shape [bag_size, 3]
        # print(coords.shape)
        # print(x.shape)
        
        # turn x to shape (batch_size, H, W, num_feat)
        x = x.unsqueeze(3).expand(batch_size, H, W, num_feat) # (batch_size, H, W, num_feat)
        # x = x.clone()
        x = _fill_abs_pos(x, y, coords, bag_indices)
        x = x.permute(0, 3, 1, 2) # (batch_size, num_feat, H, W)

        del y, adj_mat
        torch.cuda.empty_cache()

        ### SETMIL implementation from here

        # print(H, W)

        if x.device.type != 'cpu':

            # randomly sample a subset of tokens if the number of tokens is too large
            while H * W > self.max_token_num:
                if H > W:
                    # remove a row
                    idx = torch.randint(0, H, (1,)).item()
                    x = torch.cat([x[:, :, :idx], x[:, :, idx+1:]], dim=2)
                    H -= 1
                else:
                    # remove a column
                    idx = torch.randint(0, W, (1,)).item()
                    x = torch.cat([x[:, :, :, :idx], x[:, :, :, idx+1:]], dim=3)
                    W -= 1
            
        x = self.channel_reduce(x) # (batch_size, token_dim, H, W)

        if self.aspp_flag:
            x = self.aspp(x)
        else:
            x = x.reshape(batch_size, self.token_dim, -1).transpose(1,2)
            
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # pos_embed = torch.from_numpy(self.pos_embed[:, 0:x.size(1), :]).to(x.device)
        pos_embed = get_sinusoid_encoding(n_position=x.size(1), d_hid=x.size(2)).to(x.device)

        x = x + pos_embed # remove for ablation study
        x = self.pos_drop(x)
        for i in range(len(self.blocks)-1):
            blk = self.blocks[i]
            x = blk(x)
        if return_att:
            x, att = self.blocks[-1](x, return_att=return_att)
            att_abs = torch.mean(att, dim=1)[:, 0, 1:].view(batch_size, H, W) # (B, H, W)
            att_rel = torch.zeros(batch_size, bag_size).to(att_abs.device)
            att_rel = _fill_bag_seq(att_abs, att_rel, coords, bag_indices)
        else:
            x = self.blocks[-1](x)
        x = self.norm(x)[:, 0]
        x = self.head(x).squeeze(-1)
        if return_att:
            return x, att_rel
        else:
            return x

    def compute_loss(self, T_labels, X, adj_mat, *args, **kwargs):
        T_logits_pred = self.forward(X, adj_mat)
        crit_loss = self.criterion(T_logits_pred.float(), T_labels.float())
        return T_logits_pred, {self.criterion.__class__.__name__ : crit_loss}

    @torch.no_grad()
    def predict(self, X, adj_mat, *args, return_y_pred=True, **kwargs):
        """
        Input:
            X: tensor (batch_size, bag_size, ...)
        Output:
            T_logits_pred: tensor (batch_size,)
            y_pred: tensor (batch_size, bag_size) if return_y_pred is True
        """
        return self.forward(X, adj_mat, return_att=return_y_pred)   
