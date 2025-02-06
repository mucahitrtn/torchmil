import torch

from torchmil.models import MILModel


class MLP(torch.nn.Module):
    def __init__(self, in_dim, dim=512, n_layers=1):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, dim)
        self.fc_layers = torch.nn.Sequential(
            *[torch.nn.Linear(dim, dim) for _ in range(n_layers-1)]
        )
        self.act = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        if len(self.fc_layers) > 0:
            for layer in self.fc_layers:
                x = self.act(x)
                x = layer(x)
        return x
        

def build_model(config, pos_weight=None):

    ce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
    feat_ext = MLP(config.in_dim, 512, 2)
    model_config_dict = vars(config.model_config)
    
    if config.model_name == 'abmil':
        from torchmil.models import ABMIL
        return ABMIL(
            in_shape=(config.in_dim,),
            feat_ext=feat_ext,
            criterion=ce_criterion,
            **model_config_dict
        )
    # elif config.model_name == 'sm_abmil':
    #     from models import ABMIL
    #     return ABMIL(
    #         input_shape=config.data_shape,
    #         feat_ext_name=config.model_config.feat_ext_name,
    #         pool_kwargs={
    #             'att_dim': config.model_config.pool_att_dim,
    #             'sm_alpha': config.model_config.sm_alpha,
    #             'sm_mode' : config.model_config.sm_mode,
    #             'sm_steps' : config.model_config.sm_steps,
    #             'sm_where' : config.model_config.sm_where,
    #             'sm_spectral_norm' : config.model_config.sm_spectral_norm,
    #         },
    #         criterion=ce_criterion
    #     )
    # elif config.model_name == 'bayes_smooth_abmil':

    #     from models import BayesSmoothABMIL        
    #     return BayesSmoothABMIL(
    #         input_shape=config.data_shape,
    #         feat_ext_name=config.model_config.feat_ext_name,
    #         pool_kwargs={
    #             'att_dim': config.model_config.pool_att_dim,
    #             'covar_mode': config.model_config.covar_mode,
    #             # 'interactions_mode': config.model_config.interactions_mode,
    #             # 'num_heads': config.model_config.pool_transf_num_heads,
    #             # 'num_transf_layers': config.model_config.pool_transf_num_layers,
    #             # 'use_ff': config.model_config.pool_transf_use_ff,
    #             # 'dropout': config.model_config.pool_transf_dropout,
    #         },
    #         criterion=ce_criterion
    #     )
    # elif config.model_name == 'transformer_abmil':
    #     from models import TransformerABMIL
    #     return TransformerABMIL(
    #         input_shape=config.data_shape,
    #         feat_ext_name=config.model_config.feat_ext_name,
    #         transformer_encoder_kwargs={
    #             'att_dim': config.model_config.transf_att_dim,
    #             'num_heads': config.model_config.transf_num_heads,
    #             'num_layers': config.model_config.transf_num_layers,
    #             'use_ff': config.model_config.transf_use_ff,
    #             'add_self': config.model_config.transf_add_self,
    #             'dropout': config.model_config.transf_dropout
    #         },
    #         pool_kwargs={
    #             'att_dim': config.model_config.pool_att_dim
    #         },
    #         criterion=ce_criterion
    #     )
    # elif config.model_name == 'sm_transformer_abmil':
    #     from models import TransformerABMIL
    #     return TransformerABMIL(
    #         input_shape=config.data_shape,
    #         feat_ext_name=config.model_config.feat_ext_name,
    #         transformer_encoder_kwargs={
    #             'att_dim': config.model_config.transf_att_dim,
    #             'num_heads': config.model_config.transf_num_heads,
    #             'num_layers': config.model_config.transf_num_layers,
    #             'use_ff': config.model_config.transf_use_ff,
    #             'dropout': config.model_config.transf_dropout,
    #             'sm': config.model_config.sm_transformer,
    #             'sm_alpha': config.model_config.sm_alpha,
    #             'sm_mode': config.model_config.sm_mode,
    #             'sm_steps' : config.model_config.sm_steps,
    #         },
    #         pool_kwargs={
    #             'att_dim': config.model_config.pool_att_dim,
    #             'sm_alpha': config.model_config.sm_alpha,
    #             'sm_mode' : config.model_config.sm_mode,
    #             'sm_steps' : config.model_config.sm_steps,
    #             'sm_where' : config.model_config.sm_where,
    #             'sm_spectral_norm' : config.model_config.sm_spectral_norm,
    #         },
    #         criterion=ce_criterion
    #     )
    # elif config.model_name == 'transformer_bayes_smooth_abmil':
    #     from models import TransformerBayesSmoothABMIL
    #     return TransformerBayesSmoothABMIL(
    #         input_shape=config.data_shape,
    #         feat_ext_name=config.model_config.feat_ext_name,
    #         transformer_encoder_kwargs={
    #             'att_dim': config.model_config.transf_att_dim,
    #             'num_heads': config.model_config.transf_num_heads,
    #             'num_layers': config.model_config.transf_num_layers,
    #             'use_ff': config.model_config.transf_use_ff,
    #             'add_self': config.model_config.transf_add_self,
    #             'dropout': config.model_config.transf_dropout
    #         },
    #         pool_kwargs={
    #             'att_dim': config.model_config.pool_att_dim,
    #             'covar_mode': config.model_config.covar_mode,
    #         },
    #         criterion=ce_criterion
    #     )
    # elif config.model_name == 'transmil':
    #     from models import TransMIL
    #     if len(config.data_shape) > 1:
    #         raise ValueError('TransMIL only supports 1D input')
    #     return TransMIL(
    #         input_dim=config.data_shape[0],
    #         emb_dim=config.model_config.emb_dim,
    #         att_dim=config.model_config.att_dim,
    #         num_heads=config.model_config.num_heads,
    #         criterion=ce_criterion
    #         )
    # elif config.model_name == 'camil':
    #     from models import CAMIL
    #     if len(config.data_shape) > 1:
    #         raise ValueError('CAMIL only supports 1D input')
    #     return CAMIL(
    #         input_dim=config.data_shape[0],
    #         emb_dim=config.model_config.emb_dim,
    #         att_dim=config.model_config.att_dim,
    #         num_heads=config.model_config.num_heads,
    #         criterion=ce_criterion
    #         )
    # elif config.model_name == 'dsmil':
    #     from models import DSMIL
    #     return DSMIL(
    #         input_shape=config.data_shape,
    #         feat_ext_name=config.model_config.feat_ext_name,
    #         att_dim=config.model_config.att_dim,
    #         nonlinear=config.model_config.nonlinear,
    #         passing_v=config.model_config.passing_v,
    #         criterion=ce_criterion,
    #     )
    # elif config.model_name == 'gtp':
    #     from models import GTP
    #     if len(config.data_shape) > 1:
    #         raise ValueError('GTP only supports 1D input')

    #     return GTP(
    #         input_dim=config.data_shape[0], 
    #         num_layers=config.model_config.num_layers,
    #         embed_dim=config.model_config.embed_dim,
    #         num_heads=config.model_config.num_heads,
    #         node_cluster_num=config.model_config.node_cluster_num,
    #         criterion=ce_criterion
    #     )
    # elif config.model_name == 'clam':
    #     from models import CLAM_SB
    #     if len(config.data_shape) > 1:
    #         raise ValueError('CLAM only supports 1D input')
    #     return CLAM_SB(
    #         input_shape=config.data_shape,
    #         feat_ext_name=config.model_config.feat_ext_name,
    #         att_dim=config.model_config.att_dim,
    #         dropout=config.model_config.dropout,
    #         k_sample=config.model_config.k_sample,
    #         instance_loss_name=config.model_config.instance_loss_name,
    #         criterion=ce_criterion
    #         )
    # elif config.model_name == 'setmil':
    #     from models import SETMIL

    #     return SETMIL(
    #         in_chans=config.data_shape[0],
    #         num_classes=1,
    #         embed_dim=config.model_config.embed_dim,
    #         token_dim=config.model_config.token_dim,
    #         depth=config.model_config.depth,
    #         num_heads=config.model_config.num_heads, 
    #         drop_rate=config.model_config.drop_rate,
    #         attn_drop_rate=config.model_config.attn_drop_rate,
    #         drop_path_rate=config.model_config.drop_path_rate,
    #         mlp_ratio=config.model_config.mlp_ratio,
    #         qkv_bias=config.model_config.qkv_bias,
    #         aspp_flag=config.model_config.aspp_flag,
    #         criterion=ce_criterion
    #     )
    # elif 'dftdmil' in config.model_name:
    #     from models import DFTDMIL
    #     return DFTDMIL(
    #         input_shape=config.data_shape,
    #         feat_ext_name=config.model_config.feat_ext_name,
    #         att_dim=config.model_config.att_dim,
    #         num_groups=config.model_config.num_groups,
    #         distill=config.model_config.distill,
    #         criterion=ce_criterion,
    #     )
    # elif config.model_name == 'pathgcn':
    #     from models import PathGCN
    #     return PathGCN(
    #         input_dim=config.data_shape[0],
    #         num_layers=config.model_config.num_layers,
    #         resample=config.model_config.resample,
    #         hidden_dim=config.model_config.hidden_dim,
    #         criterion=ce_criterion
    #     )
    # elif config.model_name == 'deepgraphsurv':
    #     from models import DeepGraphSurv
    #     return DeepGraphSurv(
    #         input_dim=config.data_shape[0],
    #         num_layers=config.model_config.num_layers,
    #         resample=config.model_config.resample,
    #         hidden_dim=config.model_config.hidden_dim,
    #         criterion=ce_criterion
    #     )
    # elif config.model_name == 'iibmil':
    #     from models import IIBMIL
    #     return IIBMIL(
    #         input_dim=config.data_shape[0],
    #         emb_dim=config.model_config.emb_dim,
    #         depth_encoder=config.model_config.depth_encoder,
    #         depth_decoder=config.model_config.depth_decoder,
    #         num_queries=config.model_config.num_queries,
    #         criterion=ce_criterion
    #     )
    else:
        raise NotImplementedError(f'Model {config.model_name} not implemented')