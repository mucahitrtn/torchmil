import torch
import torchvision
import re
import numpy as np

def parse_model_name(model_name):
    cnn_pattern = r'cnn_(\d+)_(\d+\.\d+)' # cnn_<num_layers>_<p_dropout>
    fc_pattern = r'fc_(\d+)_(\d+)' # fc_<num_layers>_<dim>

    cnn_match = re.search(cnn_pattern, model_name)
    fc_match = re.search(fc_pattern, model_name)

    N, p, M, D = 0, 0.0, 0, 0

    if cnn_match is not None:
        N = int(cnn_match.group(1))
        p = float(cnn_match.group(2))
    if fc_match is not None:
        M = int(fc_match.group(1))
        D = int(fc_match.group(2))
    return N, p, M, D

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=2) -> None:
        super(ConvBlock, self).__init__()
        self.net = torch.nn.Sequential(
                                        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), 
                                        torch.nn.BatchNorm2d(out_channels), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=2, stride=2)
                                       )
    
    def forward(self, x):
        return self.net(x)

class Identity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return x

class MILFeatExt(torch.nn.Module):
    def __init__(self, input_shape=(3, 512, 512), feat_ext_name='cnn_7_0.3') -> None:
        super().__init__()

        self.net = None
        self.transforms = None

        if feat_ext_name == 'none':
            self.net = Identity()
        elif feat_ext_name in ['resnet18_freeze', 'resnet18_train']:
            resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
            #num_features = list(resnet.children())[-1].input_features
            if feat_ext_name == 'resnet18_freeze':
                for param in resnet.parameters():
                    param.requires_grad = False
            self.net = torch.nn.Sequential(*list(resnet.children())[:-1], torch.nn.Flatten())
            self.transforms = self.transforms = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms(antialias=True)
        elif feat_ext_name in ['mobilenetv2_freeze', 'mobilenetv2_train']:
            mobilenetv2 = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
            if feat_ext_name == 'mobilenetv2_freeze':
                for param in mobilenetv2.parameters():
                    param.requires_grad = False
            self.net = torch.nn.Sequential(*list(mobilenetv2.children())[:-1], torch.nn.AvgPool2d(7, stride=1), torch.nn.Flatten(), torch.nn.Linear(1280, 512))
            # self.transforms = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1.transforms(antialias=True)
            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(224, antialias=True), 
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif feat_ext_name in ['shufflenetv2_freeze', 'shufflenetv2_train']:
            shufflenetv2 = torchvision.models.shufflenet_v2_x0_5(weights='IMAGENET1K_V1')
            if feat_ext_name == 'shufflenetv2_freeze':
                for param in shufflenetv2.parameters():
                    param.requires_grad = False
            self.net = torch.nn.Sequential(*list(shufflenetv2.children())[:-1], torch.nn.AvgPool2d(7, stride=1), torch.nn.Flatten(), torch.nn.Linear(1024, 512))
            # self.transforms = torchvision.models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1.transforms(antialias=True)
            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(224, antialias=True), 
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            N, p, M, out_feat = parse_model_name(feat_ext_name)

            if N + M > 0:
                module_list = []

                if N > 0:
                    module_list = [ConvBlock(input_shape[0], 16, 5, 1, 2)]
                    for _ in range(N-1):
                        module_list.append(ConvBlock(16, 16, 3, 1, 0))
                        if p > 0.0:
                            module_list.append(torch.nn.Dropout(p=p))
                    module_list.append(torch.nn.Flatten())
                    

                if M > 0:
                    input_feat = np.prod(input_shape)
                    module_list = [torch.nn.Flatten(), torch.nn.Linear(input_feat, out_feat)]
                    for _ in range(M-1):
                        module_list.append(torch.nn.ReLU())
                        module_list.append(torch.nn.Linear(out_feat, out_feat))

                self.net = torch.nn.Sequential(*module_list)
            else:
                raise ValueError('Invalid model_name')

            
        self.output_size = self._get_output_size(input_shape, self.net, self.transforms)
    
    def forward(self, X):
        """
        input:
            X: tensor (batch_size, bag_size, ...)
            mask: tensor (batch_size, bag_size)
        output:
            tensor (batch_size, bag_size, output_size)
        """

        # batch_size, bag_size, C, H, W = X.shape
        # X = X.view(-1, C, H , W)
        orig_shape = X.shape
        X = X.view(-1, *orig_shape[2:])
        if self.transforms is not None:
            X = self.transforms(X)

        return self.net(X).view(*orig_shape[:2], -1) # (batch_size, bag_size, output_size)

    def _get_output_size(self, input_shape, module, transforms=None):
        """
        input:
            input_shape: tuple (C, H, W)
        output:
            output_size: int
        """
        x = torch.randn(1, *input_shape)
        if transforms is not None:
            x = transforms(x)
        return module(x).shape[-1]