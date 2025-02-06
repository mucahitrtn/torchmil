import torch
import numpy as np

from tqdm import tqdm

from torchmetrics import AUROC, MeanMetric

from copy import deepcopy

class Trainer:
    def __init__(
            self, 
            model, 
            optimizer, 
            metrics_dict = {'auroc' : AUROC(task='binary')},
            obj_metric = 'auroc',
            lr_scheduler = None, 
            annealing_scheduler_dict = None,
            device = 'cpu', 
            wandb_run = None, 
            early_stop_patience = None,
            disable_pbar = False
        ):
        self.model = model  
        self.optimizer = optimizer
        self.metrics_dict = metrics_dict
        self.obj_metric_name = obj_metric
        self.lr_scheduler = lr_scheduler
        self.annealing_scheduler_dict = annealing_scheduler_dict
        self.device = device
        self.wandb_run = wandb_run
        self.early_stop_patience = early_stop_patience
        self.disable_pbar = disable_pbar

        if self.early_stop_patience is None:
            self.early_stop_patience = np.inf
        
        self.best_model_state_dict = None
        self.best_obj_metric = None
        self.model = self.model.to(self.device)
    
    def _log(self, metrics):
        if self.wandb_run is not None:
            self.wandb_run.log(metrics)
    
    def _get_model_state_dict(self):
        state_dict = deepcopy(self.model.state_dict())
        return state_dict

    def train(self, max_epochs, train_dataloader, val_dataloader=None, test_dataloader=None):

        if val_dataloader is None:
            val_dataloader = train_dataloader
        
        if self.best_model_state_dict is None:
            self.best_model_state_dict = self._get_model_state_dict()
            self.best_obj_metric = -np.inf

        early_stop_count = 0
        for epoch in range(1, max_epochs+1):

            # Train loop
            train_metrics = self._shared_loop(train_dataloader, disable_pbar=self.disable_pbar, epoch=epoch, mode='train')
            self._log(train_metrics)
            torch.cuda.empty_cache() # clear cache

            # Validation loop
            val_metrics = self._shared_loop(val_dataloader, disable_pbar=self.disable_pbar, epoch=epoch, mode='val')

            self._log(val_metrics)
            torch.cuda.empty_cache() # clear cache
            
            # Test loop
            if test_dataloader is not None:
                test_metrics = self._shared_loop(test_dataloader, disable_pbar=self.disable_pbar, epoch=epoch, mode='test')
                self._log(test_metrics)
                torch.cuda.empty_cache() # clear cache
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                
            print(f'Best {self.obj_metric_name}: {self.best_obj_metric}, Current {self.obj_metric_name}: {val_metrics[f"val/{self.obj_metric_name}"]}')
            if val_metrics[f"val/{self.obj_metric_name}"] < self.best_obj_metric:
                early_stop_count += 1
                print(f'Early stopping count: {early_stop_count}')
            else:
                self.best_obj_metric = val_metrics[f"val/{self.obj_metric_name}"]
                self.best_model_state_dict = self._get_model_state_dict()
                early_stop_count = 0
            
            if early_stop_count >= self.early_stop_patience:
                print(f'Reached early stopping condition')
                break


    def _shared_loop(self, dataloader, disable_pbar=False, epoch=0, mode='train'):
        
        if mode=='train':
            name = 'Train'
        elif mode=='val':
            name = 'Validation'
        elif mode=='test':
            name = 'Test'

        self.model.train()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), disable=disable_pbar)
        pbar.set_description(f"[Epoch {epoch}] {name} ")

        loop_metrics_dict = self.metrics_dict
        for k in loop_metrics_dict.keys():
            loop_metrics_dict[k].reset()
        loop_loss_dict = {'loss' : MeanMetric()}

        for batch_idx, batch in pbar:
            
            Y = batch['Y'].to(self.device) # (batch_size, 1)

            self.optimizer.zero_grad()

            Y_pred, loss_dict = self.model.compute_loss(**batch)

            loss = 0.0
            for loss_name, loss_value in loss_dict.items():
                coef = 1.0
                if self.annealing_scheduler_dict is not None:
                    if loss_name in self.annealing_scheduler_dict.keys():
                        coef = self.annealing_scheduler_dict[loss_name]()
                loss += coef*loss_value
                if loss_name not in loop_loss_dict.keys():
                    loop_loss_dict[loss_name] = MeanMetric()
                loop_loss_dict[loss_name].update(loss_value.item())
            loop_loss_dict['loss'].update(loss.item())

            if mode=='train':

                loss.backward()
                self.optimizer.step()

                if self.annealing_scheduler_dict is not None:
                    for annealing_scheduler in self.annealing_scheduler_dict.values():
                        annealing_scheduler.step()

            for k in loop_metrics_dict.keys():
                loop_metrics_dict[k].update(Y_pred, Y)

            if batch_idx < (len(dataloader) - 1):
                pbar.set_postfix({f'{mode}/{loss_name}' : loop_loss_dict[loss_name].compute().item()  for loss_name in loss_dict})
            else:
                train_metrics = {f'{mode}/{k}' : v.compute().item() for k,v in loop_loss_dict.items()}
                train_metrics = {**train_metrics, **{f'{mode}/{k}' : v.compute().item() for k,v in loop_metrics_dict.items()}}              
                pbar.set_postfix(train_metrics)
            
            del Y, Y_pred, loss
        pbar.close()
        return train_metrics
    
    def get_best_model_state_dict(self):
        return self.best_model_state_dict
