import torch

def build_MIL_model(args, pos_weight=None):

    ce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
    
    if args.model_name == 'abmil':
        from models import ABMIL
        return ABMIL(
            input_shape=args.data_shape,
            feat_ext_name=args.model_config.feat_ext_name,
            pool_kwargs={
                'att_dim': args.model_config.pool_att_dim
            },
            criterion=ce_criterion
        )
    elif args.model_name == 'sm_abmil':
        from models import ABMIL
        return ABMIL(
            input_shape=args.data_shape,
            feat_ext_name=args.model_config.feat_ext_name,
            pool_kwargs={
                'att_dim': args.model_config.pool_att_dim,
                'sm_alpha': args.model_config.sm_alpha,
                'sm_mode' : args.model_config.sm_mode,
                'sm_steps' : args.model_config.sm_steps,
                'sm_where' : args.model_config.sm_where,
                'sm_spectral_norm' : args.model_config.sm_spectral_norm,
            },
            criterion=ce_criterion
        )
    elif args.model_name == 'bayes_smooth_abmil':

        from models import BayesSmoothABMIL        
        return BayesSmoothABMIL(
            input_shape=args.data_shape,
            feat_ext_name=args.model_config.feat_ext_name,
            pool_kwargs={
                'att_dim': args.model_config.pool_att_dim,
                'covar_mode': args.model_config.covar_mode,
                # 'interactions_mode': args.model_config.interactions_mode,
                # 'num_heads': args.model_config.pool_transf_num_heads,
                # 'num_transf_layers': args.model_config.pool_transf_num_layers,
                # 'use_ff': args.model_config.pool_transf_use_ff,
                # 'dropout': args.model_config.pool_transf_dropout,
            },
            criterion=ce_criterion
        )
    elif args.model_name == 'transformer_abmil':
        from models import TransformerABMIL
        return TransformerABMIL(
            input_shape=args.data_shape,
            feat_ext_name=args.model_config.feat_ext_name,
            transformer_encoder_kwargs={
                'att_dim': args.model_config.transf_att_dim,
                'num_heads': args.model_config.transf_num_heads,
                'num_layers': args.model_config.transf_num_layers,
                'use_ff': args.model_config.transf_use_ff,
                'add_self': args.model_config.transf_add_self,
                'dropout': args.model_config.transf_dropout
            },
            pool_kwargs={
                'att_dim': args.model_config.pool_att_dim
            },
            criterion=ce_criterion
        )
    elif args.model_name == 'sm_transformer_abmil':
        from models import TransformerABMIL
        return TransformerABMIL(
            input_shape=args.data_shape,
            feat_ext_name=args.model_config.feat_ext_name,
            transformer_encoder_kwargs={
                'att_dim': args.model_config.transf_att_dim,
                'num_heads': args.model_config.transf_num_heads,
                'num_layers': args.model_config.transf_num_layers,
                'use_ff': args.model_config.transf_use_ff,
                'dropout': args.model_config.transf_dropout,
                'sm': args.model_config.sm_transformer,
                'sm_alpha': args.model_config.sm_alpha,
                'sm_mode': args.model_config.sm_mode,
                'sm_steps' : args.model_config.sm_steps,
            },
            pool_kwargs={
                'att_dim': args.model_config.pool_att_dim,
                'sm_alpha': args.model_config.sm_alpha,
                'sm_mode' : args.model_config.sm_mode,
                'sm_steps' : args.model_config.sm_steps,
                'sm_where' : args.model_config.sm_where,
                'sm_spectral_norm' : args.model_config.sm_spectral_norm,
            },
            criterion=ce_criterion
        )
    elif args.model_name == 'transformer_bayes_smooth_abmil':
        from models import TransformerBayesSmoothABMIL
        return TransformerBayesSmoothABMIL(
            input_shape=args.data_shape,
            feat_ext_name=args.model_config.feat_ext_name,
            transformer_encoder_kwargs={
                'att_dim': args.model_config.transf_att_dim,
                'num_heads': args.model_config.transf_num_heads,
                'num_layers': args.model_config.transf_num_layers,
                'use_ff': args.model_config.transf_use_ff,
                'add_self': args.model_config.transf_add_self,
                'dropout': args.model_config.transf_dropout
            },
            pool_kwargs={
                'att_dim': args.model_config.pool_att_dim,
                'covar_mode': args.model_config.covar_mode,
            },
            criterion=ce_criterion
        )
    elif args.model_name == 'transmil':
        from models import TransMIL
        if len(args.data_shape) > 1:
            raise ValueError('TransMIL only supports 1D input')
        return TransMIL(
            input_dim=args.data_shape[0],
            emb_dim=args.model_config.emb_dim,
            att_dim=args.model_config.att_dim,
            num_heads=args.model_config.num_heads,
            criterion=ce_criterion
            )
    elif args.model_name == 'camil':
        from models import CAMIL
        if len(args.data_shape) > 1:
            raise ValueError('CAMIL only supports 1D input')
        return CAMIL(
            input_dim=args.data_shape[0],
            emb_dim=args.model_config.emb_dim,
            att_dim=args.model_config.att_dim,
            num_heads=args.model_config.num_heads,
            criterion=ce_criterion
            )
    elif args.model_name == 'dsmil':
        from models import DSMIL
        return DSMIL(
            input_shape=args.data_shape,
            feat_ext_name=args.model_config.feat_ext_name,
            att_dim=args.model_config.att_dim,
            nonlinear=args.model_config.nonlinear,
            passing_v=args.model_config.passing_v,
            criterion=ce_criterion,
        )
    elif args.model_name == 'gtp':
        from models import GTP
        if len(args.data_shape) > 1:
            raise ValueError('GTP only supports 1D input')

        return GTP(
            input_dim=args.data_shape[0], 
            num_layers=args.model_config.num_layers,
            embed_dim=args.model_config.embed_dim,
            num_heads=args.model_config.num_heads,
            node_cluster_num=args.model_config.node_cluster_num,
            criterion=ce_criterion
        )
    elif args.model_name == 'clam':
        from models import CLAM_SB
        if len(args.data_shape) > 1:
            raise ValueError('CLAM only supports 1D input')
        return CLAM_SB(
            input_shape=args.data_shape,
            feat_ext_name=args.model_config.feat_ext_name,
            att_dim=args.model_config.att_dim,
            dropout=args.model_config.dropout,
            k_sample=args.model_config.k_sample,
            instance_loss_name=args.model_config.instance_loss_name,
            criterion=ce_criterion
            )
    elif args.model_name == 'setmil':
        from models import SETMIL

        return SETMIL(
            in_chans=args.data_shape[0],
            num_classes=1,
            embed_dim=args.model_config.embed_dim,
            token_dim=args.model_config.token_dim,
            depth=args.model_config.depth,
            num_heads=args.model_config.num_heads, 
            drop_rate=args.model_config.drop_rate,
            attn_drop_rate=args.model_config.attn_drop_rate,
            drop_path_rate=args.model_config.drop_path_rate,
            mlp_ratio=args.model_config.mlp_ratio,
            qkv_bias=args.model_config.qkv_bias,
            aspp_flag=args.model_config.aspp_flag,
            criterion=ce_criterion
        )
    elif 'dftdmil' in args.model_name:
        from models import DFTDMIL
        return DFTDMIL(
            input_shape=args.data_shape,
            feat_ext_name=args.model_config.feat_ext_name,
            att_dim=args.model_config.att_dim,
            num_groups=args.model_config.num_groups,
            distill=args.model_config.distill,
            criterion=ce_criterion,
        )
    elif args.model_name == 'pathgcn':
        from models import PathGCN
        return PathGCN(
            input_dim=args.data_shape[0],
            num_layers=args.model_config.num_layers,
            resample=args.model_config.resample,
            hidden_dim=args.model_config.hidden_dim,
            criterion=ce_criterion
        )
    elif args.model_name == 'deepgraphsurv':
        from models import DeepGraphSurv
        return DeepGraphSurv(
            input_dim=args.data_shape[0],
            num_layers=args.model_config.num_layers,
            resample=args.model_config.resample,
            hidden_dim=args.model_config.hidden_dim,
            criterion=ce_criterion
        )
    elif args.model_name == 'iibmil':
        from models import IIBMIL
        return IIBMIL(
            input_dim=args.data_shape[0],
            emb_dim=args.model_config.emb_dim,
            depth_encoder=args.model_config.depth_encoder,
            depth_decoder=args.model_config.depth_decoder,
            num_queries=args.model_config.num_queries,
            criterion=ce_criterion
        )
    else:
        raise NotImplementedError(f'Model {args.model_name} not implemented')