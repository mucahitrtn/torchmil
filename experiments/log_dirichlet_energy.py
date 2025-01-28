

# CUDA_VISIBLE_DEVICES=0 nohup python code/experiments/log_dirichlet_energy.py --dataset_name=rsna-features_resnet18 --model_name=abmil > rsna_dir_energy_abmil.out 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python code/experiments/log_dirichlet_energy.py --dataset_name=rsna-features_resnet18 --model_name=transformer_abmil > rsna_dir_energy_transabmil.out 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python code/experiments/log_dirichlet_energy.py --dataset_name=panda-patches_512-features_resnet18 --model_name=abmil > panda_dir_energy_abmil.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python code/experiments/log_dirichlet_energy.py --dataset_name=panda-patches_512-features_resnet18 --model_name=transformer_abmil > panda_dir_energy_transabmil.out 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python code/experiments/log_dirichlet_energy.py --dataset_name=camelyon16-patches_512_preset-features_resnet50_bt --model_name=abmil > camelyon_dir_energy_abmil.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python code/experiments/log_dirichlet_energy.py --dataset_name=camelyon16-patches_512_preset-features_resnet50_bt --model_name=transformer_abmil > camelyon_dir_energy_transabmil.out 2>&1 &



import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["WANDB_MODE"] = "online"

import sys
sys.path.append('/work/work_fran/SmoothAttention/code')

import torch
import wandb
import argparse
from tqdm import tqdm

from utils import seed_everything, MIL_collate_fn
from model_builder import build_MIL_model
from dataset_loader import load_test_dataset
from config import correct_args


print(torch.__version__)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', default='', type=str, help="Target dataset")
    parser.add_argument('--model_name', default='', type=str, help="Target model")

    args = parser.parse_args()

    return args

def compute_dir_energy(model, dataloader, device = 'cuda'):

    model = model.to(device)
    model.eval()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    pbar.set_description("Test")

    dir_energy_sum = 0
    dir_energy_norm_sum = 0
    n_batches = 0
    # with torch.no_grad():
    for bag_idx, batch in pbar:

        batch_size = batch[0].shape[0]
        bag_size = batch[0].shape[1]

        if batch_size != 1:
            raise ValueError("[predict] Batch size must be 1")

        X, T, y, adj_mat, mask = batch # X: (batch_size, bag_size, 3, 512, 512), T: (batch_size, bag_size), y: (batch_size, 1), adj_mat: (batch_size, bag_size, bag_size), mask: (batch_size, bag_size)
        X = X.to(device)
        T = T.to(device)
        y = y.to(device)
        adj_mat = adj_mat.to(device)
        mask = mask.to(device)

        T_logits_pred, f_pred = model.predict(X, adj_mat, mask, return_y_pred=True) # T_logits_pred: (batch_size,), f_pred: (batch_size, bag_size)

        if adj_mat.is_sparse:
            adj_mat = adj_mat.to_dense()

        f_pred = f_pred.unsqueeze(-1) # (batch_size, bag_size, 1)

        A_f = torch.bmm(adj_mat, f_pred) # (batch_size, bag_size, 1)
        fT_A_f = torch.bmm(f_pred.transpose(1, 2), A_f) # (batch_size, 1, 1)
        fT_f = torch.bmm(f_pred.transpose(1, 2), f_pred) # (batch_size, 1, 1)

        fT_L_f = fT_f - fT_A_f # (batch_size, 1, 1)
        

        fT_f = fT_f.squeeze()
        fT_L_f = fT_L_f.squeeze()
    
        dir_energy = 0.5 * fT_L_f 
        dir_energy_norm = dir_energy / fT_f

        dir_energy_sum += dir_energy.item()
        dir_energy_norm_sum += dir_energy_norm.item()
        n_batches += 1
    
    dir_energy = dir_energy_sum / n_batches
    dir_energy_norm = dir_energy_norm_sum / n_batches

    return dir_energy, dir_energy_norm

def main():

    args = parse_args()

    api = wandb.Api()
    # runs = api.runs("francastro-team/SmoothAttention", {"tags": "relevant"})
    # runs = api.runs("francastro-team/SmoothAttention")

    runs = api.runs(
        "francastro-team/SmoothAttention", 
        {
            "config.dataset_name": args.dataset_name,
            "config.model_name": args.model_name
        }
    )

    for run in runs:
        config = run.config
        config = argparse.Namespace(**config)
        config = correct_args(config)

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

        print('Run config:')
        for arg in vars(config):
            print('{:<25s}: {:s}'.format(arg, str(getattr(config, arg))))

        seed_everything(config.seed)

        test_dataset = load_test_dataset(config)

        collate_fn = lambda x: MIL_collate_fn(x, use_sparse=config.use_sparse)

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=config.num_workers, 
            sampler=None, 
            collate_fn=collate_fn
        )

        if config.test_in_cpu:
            evaluate_device = 'cpu'
        else:
            evaluate_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = build_MIL_model(config)
        weights = run.file('weights/best.pt').download(f'./weights/tmp/{args.dataset_name}-{args.model_name}/', replace=True)
        weights_dict = torch.load(weights.name)
        model.load_state_dict(weights_dict, strict=False)

        dir_energy, dir_energy_norm = compute_dir_energy(model, test_dataloader, evaluate_device)

        print(f"Dirichlet energy: {dir_energy}")
        print(f"Dirichlet energy normalized: {dir_energy_norm}")

        wandb.log({'test/dir_energy': dir_energy, 'test/dir_energy_norm': dir_energy_norm})

        print(f"Run {run.name} finished!")    
        print('--------------------------------------------------------------------------------')
        wandb.finish()
    print('Done!')

if __name__ == "__main__":
    main()