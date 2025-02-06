import argparse
import yaml

def alpha_type(value):
    if value in ['trainable', 'estimate']:
        return value
    else:
        return float(value)

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode', 
        default='train_test', 
        type=str, 
        help="Mode to run the code (train_test/train/test)"
    )

    parser.add_argument(
        '--use_wandb',
        action='store_true', 
        help="Use wandb or not"
    )

    parser.add_argument(
        '--wandb_project', 
        default='MIL-benchmark', 
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
        '--test_in_cpu', 
        action='store_true', 
        help="Test in cpu"
    )

    parser.add_argument(
        '--use_sparse', 
        action='store_true', 
        help="Use sparse tensors to store the adjacency matrix"
    )

    parser.add_argument(
        '--weights_dir', 
        default='./weights/', 
        type=str, 
        metavar='PATH', 
        help="Path to save the model weights"
    )

    parser.add_argument(
        '--results_dir', 
        default='./results/', 
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
        default='./config_files/rsna_config.yml', 
        help="Config file to load the model settings"
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
        
    return args