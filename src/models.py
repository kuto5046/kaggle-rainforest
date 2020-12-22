import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import torchvision
from src.dataset import SpectrogramDataset
# import src.configuration as C
from resnest.torch import resnest50

class ResNet50(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(ResNet50, self).__init__()

        model = torchvision.models.resnet50(pretrained=pretrained)
        layers = list(model.children())[:-2]  # 後ろ２層を除く(adaptiveAvgpooling(1,1)とlinear)
        layers.append(nn.AdaptiveMaxPool2d(1))  # 後ろに追加 この処理で(1,1) → 1に変換している
        self.encoder = nn.Sequential(*layers)  # 1chで出力するencoder

        in_features = model.fc.in_features  # 最終層の入力ch

        # 最終層のch数合わせ(１層で一気に行ってはだめ？)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes))


    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x


class ResNeSt50(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(ResNeSt50, self).__init__()

        self.model = resnest50(pretrained=pretrained)
        del self.model.fc
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def get_model(config):
    model_config = config["model"]
    model_name = model_config["name"]
    model_params = model_config["params"]

    if model_name == "ResNet50":
        model = ResNet50(**model_params)
        return model
    elif model_name == "ResNeSt50":
        model = ResNeSt50(**model_params)
        return model
    else:
        raise NotImplementedError

