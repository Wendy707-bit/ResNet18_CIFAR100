import torch
from torch import nn
import torchvision
from torchvision import models


def build_model (device):
    #加载ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    
    in_features = model.fc.in_features#获取原最后一层的输入维度
    model.fc = nn.Linear(in_features, 100)#替换最后一层

    #把模型搬到GPU或者CPU上
    model=model.to(device)
    return model

#打印模型结构
def print_model_structure(model,device):
    from torchsummary import summary
    summary(model, input_size=(3, 224, 224), device=device)
