import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["WANDB_MODE"] = "online"

import sys
sys.path.append('/work/work_fran/SmoothAttention/code')

import torch
import wandb
import argparse

from utils import seed_everything
from train_test import train_test


print(torch.__version__)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', default='', type=str, help="Target dataset")

    args = parser.parse_args()

    return args

def main():

    args = parse_args()
    dataset_name = args.dataset_name

    api = wandb.Api()
    # runs = api.runs("francastro-team/SmoothAttention", {"tags": "relevant"})
    runs = api.runs("francastro-team/SmoothAttention")

    # filter runs 
    if dataset_name != '':
        runs = [run for run in runs if run.state == 'finished']
        runs = [run for run in runs if run.config['dataset_name'] == dataset_name]
        # runs = [run for run in runs if run.config['model_name'] != 'bayes_smooth_abmil']
        # runs = [run for run in runs if 'test/bag/auprc' not in run.summary.keys()]
    
    print(f'Found {len(runs)} runs for dataset {dataset_name}.')

    for run in runs:
        config = run.config
        config = argparse.Namespace(**config)

        # get run id
        run_id = run.id
            
        wandb.init(
            id=run_id, 
            project='SmoothAttention', 
            resume="must", 
            reinit=True,
            mode = "online"
        )
        
        print('--------------------------------------------------------------------------------')
        print(f'Processing run {run.name}...')
        # config.load_weights_path = 'tmp/weights/best.pt'
        run.file('weights/best.pt').download(replace=True, root=f'/tmp/francastro-team/{run_id}/')
        config.load_weights_path = f'/tmp/francastro-team/{run_id}/weights/best.pt'
        # wandb.run = None

        print('Run config:')
        for arg in vars(config):
            print('{:<25s}: {:s}'.format(arg, str(getattr(config, arg))))

        seed_everything(config.seed)

        run_train = False
        run_test = True

        train_test(config, run_train, run_test)

        wandb.run.finish()
        print(f'Run {run.name} finished!')    
    print('Done!')

if __name__ == "__main__":
    main()