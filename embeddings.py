import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch.hub import load_state_dict_from_url
import torchvision


class BaseImageEmbedding(nn.Module):
    def __init__(self, model, img_res=224):
        super().__init__()

        self.model = model
        self.img_res = img_res
        self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                      requires_grad=False)
        self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                     requires_grad=False)

    def forward(self, x):
        # expects image in range [-1, 1]
        x = (x + 1.) / 2.0
        x = (x - self.mean) / self.std

        if (x.shape[2] != self.img_res) or (x.shape[3] != self.img_res):
            x = F.interpolate(x,
                              size=(self.img_res, self.img_res),
                              mode='bilinear',
                              align_corners=True)

        x = self.model(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class Inceptionv3Embedding(BaseImageEmbedding):
    def __init__(self):
        model = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        model.fc = Identity()
        super().__init__(model, img_res=299)


class ResNet50Embedding(BaseImageEmbedding):
    def __init__(self):
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = Identity()
        super().__init__(model)


class Places365Embedding(BaseImageEmbedding):
    def __init__(self, arch='resnet50'):
        # arch can be alexnet, resnet18, resnet50, or densenet161
        weights_url = 'http://places2.csail.mit.edu/models_places365/{}_places365.pth.tar'.format(arch)
        state_dict = load_state_dict_from_url(weights_url, progress=True)
        state_dict = {str.replace(k,'module.',''): v for k,v in state_dict['state_dict'].items()}

        model = torchvision.models.__dict__[arch](num_classes=365)
        model.load_state_dict(state_dict)
        model.fc = Identity()
        super().__init__(model)


class ResNextWSL(BaseImageEmbedding):
    def __init__(self, d=8):
        model = torch.hub.load('facebookresearch/WSL-Images',
                               'resnext101_32x{}d_wsl'.format(d))
        model.fc = Identity()
        super().__init__(model)


class SwAVEmbedding(BaseImageEmbedding):
    def __init__(self):
        model = torch.hub.load('facebookresearch/swav', 'resnet50')
        model.fc = Identity()
        super().__init__(model)
