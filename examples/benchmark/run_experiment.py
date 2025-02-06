import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import wandb

from utils import seed_everything
from arguments import get_arguments
from train_test import train_test

print(torch.__version__)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

def main():

    config = get_arguments()

    if config.use_wandb:
        wandb_config = config.__dict__
        for k in wandb_config.keys():
            if wandb_config[k] is None:
                wandb_config[k] = 'None'
        wandb.init(project=config.wandb_project, config=wandb_config)
        config.save_weights_path = config.load_weights_path = wandb.run.dir + '/weights/best.pt'
        wandb_run = wandb.run
    else:
        config.save_weights_path = config.load_weights_path = None
        wandb_run = None
    setattr(config, 'wandb_run', wandb_run)

    print('Arguments:')
    for arg in vars(config):
        print('{:<25s}: {:s}'.format(arg, str(getattr(config, arg))))

    seed_everything(config.seed)

    run_train = 'train' in config.mode
    run_test = 'test' in config.mode

    train_test(config, run_train, run_test)
    
    print('Done!')

if __name__ == "__main__":
    main()
