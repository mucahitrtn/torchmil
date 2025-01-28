import argparse
import yaml

def alpha_type(value):
    if value in ['trainable', 'estimate']:
        return value
    else:
        return float(value)

def correct_args(args):

    if args.model_name not in ['transformer_bayes_smooth_abmil', 'bayes_smooth_abmil']:
        args.covar_mode = None
        args.interactions_mode = None

    if args.model_name not in ['sm_abmil', 'sm_transformer_abmil']:
        args.sm_alpha = None
        args.sm_steps = None
        args.sm_mode = None
        args.sm_where = None
        args.sm_spectral_norm = None

    if args.sm_alpha in [0.0, None]:
        args.sm_mode = None
        args.sm_where = None
        args.sm_steps = None

    if args.model_name in ['setmil']:
        setattr(args, 'adj_mat_mode', 'absolute')
        args.use_sparse = True
    else:
        setattr(args, 'adj_mat_mode', 'relative')
    
    return args

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode', 
        default='train_test', 
        type=str, 
        help="Mode to run the code (train/test)"
    )

    parser.add_argument(
        '--use_wandb',
        action='store_true', 
        help="Use wandb or not"
    )

    parser.add_argument(
        '--wandb_project', 
        default='SmoothAttention-bayes', 
        type=str, help="Wandb project name"
    )
    
    parser.add_argument(
        '--num_workers', 
        default=12, 
        type=int, 
        help="Number of workers to load data"
    )

    parser.add_argument(
        '--pin_memory', 
        action='store_true', 
        help="Pin memory or not"
    )

    parser.add_argument(
        '--distributed', 
        action='store_true', 
        help="Use distributed training"
    )

    parser.add_argument(
        '--test_in_cpu', 
        action='store_true', 
        help="Test in cpu"
    )

    parser.add_argument(
        '--use_sparse', 
        action='store_true', 
        help="Use sparse tensors to store the adjacency matrix"
    )

    # Path settings
    parser.add_argument(
        '--history_dir', 
        default='/work/work_fran/SmoothAttention/history/', 
        type=str, 
        metavar='PATH', 
        help="Path to save the history file"
    )

    parser.add_argument(
        '--weights_dir', 
        default='/work/work_fran/SmoothAttention/weights/', 
        type=str, 
        metavar='PATH', 
        help="Path to save the model weights"
    )

    parser.add_argument(
        '--results_dir', 
        default='results/', 
        type=str, 
        metavar='PATH', 
        help="Path to save the results"
    )

    # Experiment settings
    parser.add_argument(
        '--seed', 
        type=int, 
        default=0, 
        help="Seed"
    )

    parser.add_argument(
        '--dataset_name', 
        default='rsna-features_resnet18', 
        type=str, 
        help="Dataset to use"
    )

    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=4, 
        help="Batch size of training"
    )

    parser.add_argument(
        '--val_prop', 
        type=float, 
        default=0.2, 
        help="Proportion of validation data"
    )

    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50, 
        help="Training epochs"
    )

    parser.add_argument(
        '--config_file', 
        type=str, 
        default='/work/work_fran/SmoothAttention/code/experiments/config_files/rsna_config.yml', 
        help="Config file to load the settings"
    )

    parser.add_argument(
        '--use_inst_distances', 
        action='store_true', 
        help="Use instance distances or not to build the adjacency matrix"
    )

    # Model settings
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='abmil', 
        help="Model name"
    )

    # Training settings
    parser.add_argument(
        '--lr', 
        type=float, 
        default=1e-4, 
        help="Initial learning rate"
    )

    parser.add_argument(
        '--patience', 
        type=int, 
        default=10, 
        help="Patience for early stopping"
    )

    parser.add_argument(
        '--weight_decay', 
        type=float, 
        default=0.0, 
        help="Weight decay for the optimizer"
    )

    # Sm hyperparams
    parser.add_argument(
        '--sm_alpha', 
        type=alpha_type, 
        default='trainable', 
        help="\\alpha for the Sm operator"
    )

    parser.add_argument(
        '--sm_mode', 
        type=str, 
        default='approx', 
        help="Mode for the Sm operator"
    )

    parser.add_argument(
        '--sm_steps', 
        type=int, 
        default=10, 
        help="Number of steps to approximate the Sm operator"
    )

    parser.add_argument(
        '--sm_where', 
        type=str, 
        default='mid', 
        help="Where to place the Sm operator within the attention pool"
    )

    parser.add_argument(
        '--sm_transformer', 
        action='store_true', 
        help="Whether to use the Sm operator in the transformer encoder"
    )

    parser.add_argument(
        '--sm_spectral_norm', 
        action='store_true', 
        help="Use spectral normalization or not"
    )

    # Bayes Smooth hyperparams
    parser.add_argument(
        '--covar_mode', 
        type=str, 
        default='diag', 
        help="Covariance mode for the bayes smooth attention"
    )

    parser.add_argument(
        '--annealing_min_coef', 
        type=float, 
        default=0.0, 
        help="Minimum coefficient for the annealing scheduler"
    )

    parser.add_argument(
        '--annealing_max_coef', 
        type=float, 
        default=1.0, 
        help="Maximum coefficient for the annealing scheduler"
    )

    args = parser.parse_args()

    # Add model config to args
    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
            yaml_dict = yaml.safe_load(f)
        
        for key in yaml_dict.keys():
            if key != 'model_config':
                setattr(args, key, yaml_dict[key])

        model_config_dict = yaml_dict['model_config'][args.model_name]
        args.model_config = argparse.Namespace()
        for key in model_config_dict.keys():
            
            # if key exists in args, set the value from the command line, otherwise set the value from the config file
            if hasattr(args, key):
                setattr(args.model_config, key, getattr(args, key))
            else:
                setattr(args.model_config, key, model_config_dict[key])
    
    args = correct_args(args)    
    
    return args