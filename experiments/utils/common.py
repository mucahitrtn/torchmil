import torch
import os
import torchvision    
import numpy as np
import random

def load_torchvision_model(model_name, use_imagenet_pretrained_weights=True):
    model = None
    transforms = None
    if model_name == 'mobilenet_v2':
        if use_imagenet_pretrained_weights:
            weights = 'IMAGENET1K_V2'
        else:
            weights = None
        model = torchvision.models.mobilenet_v2(weights = weights)
        model = torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten())
        # transforms = torchvision.transforms.Compose([
        #     torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1.transforms(antialias=True)
        # ])
        transforms = torchvision.transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )
        # n_feat = 1280
    elif model_name == 'mobilenet_v3_large':
        if use_imagenet_pretrained_weights:
            weights = 'IMAGENET1K_V2'
        else:
            weights = None
        model = torchvision.models.mobilenet_v3_large(weights = weights)
        model = torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten())
        # transforms = torchvision.transforms.Compose([
        #     torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1.transforms(antialias=True)
        # ])
        transforms = torchvision.transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )
        # n_feat = 960
    elif model_name == 'resnet18':
        if use_imagenet_pretrained_weights:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.resnet18(weights = weights)
        model = torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten())
        # transforms = torchvision.transforms.Compose([
        #     torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms(antialias=True)
        # ])
        transforms = torchvision.transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )
        # n_feat = 512
    elif model_name == 'resnet50':
        if use_imagenet_pretrained_weights:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.resnet50(weights = weights)
        model = torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten())
        # transforms = torchvision.transforms.Compose([
        #     torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms(antialias=True)
        # ])
        transforms = torchvision.transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )
    else:
        raise NotImplementedError
    
    return model, transforms

def get_local_rank():
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = 0
    return local_rank

def ddp_setup():
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)