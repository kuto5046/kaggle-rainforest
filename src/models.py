import os
import random
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.distributions import Beta

import pytorch_lightning as pl
import torchvision
from resnest.torch import resnest50
from sklearn.metrics import accuracy_score
from src.dataset import SpectrogramDataset
import src.configuration as C
from src.metric import LWLRAP
import pytorch_lightning as pl
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


def calc_acc(pred, y):
    pred = torch.sigmoid(pred).detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    return accuracy_score(y, pred)


# Learner class(pytorch-lighting)
class Learner(pl.LightningModule):
    def __init__(self, config, fold):
        super().__init__()
        self.config = config
        self.fold = fold

    def forward(self, x):
        return None
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        p = random.random()
        do_mixup = True if p < self.config['mixup']['prob'] else False

        if self.config['mixup']['flag'] and do_mixup:
            x, y, y_shuffle, lam = mixup_data(x, y, alpha=self.config['mixup']['alpha'])

        pred = self.forward(x)
        criterion = C.get_criterion(self.config)
        if self.config['mixup']['flag'] and do_mixup:
            loss = mixup_criterion(criterion, pred, y, y_shuffle, lam)
        else:
            loss = criterion(pred, y)
        lwlrap = LWLRAP(pred, y)
        # acc = calc_acc(pred, y)
        self.log(f'loss/train', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'LWLRAP/train', lwlrap, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log(f'fold{self.fold}_train_acc', acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss
    

    # batchのxはlist型
    def validation_step(self, batch, batch_idx):
        # xが複数の場合
        x_list, y = batch
        batch_size = x_list.shape[0]
        x = x_list.view(-1, x_list.shape[2], x_list.shape[3], x_list.shape[4])  # batch>1でも可
        output = self.forward(x)
        output = output.view(batch_size, -1, y.shape[1])  # y.shape[1]==num_classes
        pred = torch.max(output, dim=1)[0]  # 1次元目(分割sしたやつ)で各クラスの最大を取得

        """
        # xが１つの場合
        x, y = batch
        output = self.model(x)
        pred = output[self.config["globals"]["output_type"]]
        """
        criterion = C.get_criterion(self.config)
        loss = criterion(pred, y)
        lwlrap = LWLRAP(pred, y)
        # acc = calc_acc(pred, y)
        self.log(f'loss/val', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'LWLRAP/val', lwlrap, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log(f'fold{self.fold}_val_acc', acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)  
        return loss

    def configure_optimizers(self):
        optimizer = C.get_optimizer(self.model, self.config)
        scheduler = C.get_scheduler(optimizer, self.config)
        return [optimizer], [scheduler]


class ResNet50Learner(Learner):
    def __init__(self, config, fold):
        super().__init__(config, fold)
        self.pretrained = config['model']['params']['pretrained']
        self.num_classes = config['model']['params']['num_classes']

        self.model = torchvision.models.resnet50(pretrained=self.pretrained)
        layers = list(self.model.children())[:-2]  # 後ろ２層を除く(adaptiveAvgpooling(1,1)とlinear)
        layers.append(nn.AdaptiveMaxPool2d(1))  # 後ろに追加 この処理で(1,1) → 1に変換している
        self.encoder = nn.Sequential(*layers)  # 1chで出力するencoder

        in_features = self.model.fc.in_features  # 最終層の入力ch

        # 最終層のch数合わせ(１層で一気に行ってはだめ？)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024,self.num_classes))


    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x


class ResNeSt50Learner(Learner):
    def __init__(self, config, fold):
        super().__init__(config, fold)
        self.pretrained = config['model']['params']['pretrained']
        self.num_classes = config['model']['params']['num_classes']

        self.model = resnest50(pretrained=self.pretrained)
        del self.model.fc
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, self.num_classes)
        )

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2)

    def forward(self, x):
        """
        # spec aug
        if self.training:
            x = self.spec_augmenter(x)
        """
        x = self.model(x)
        return x


def get_model(config, fold):
    model_name = config["model"]["name"]
    if model_name == "ResNet50":
        model = ResNet50Learner(config, fold)
        return model
    elif model_name == "ResNeSt50":
        model = ResNeSt50Learner(config, fold)
        return model
    else:
        raise NotImplementedError


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)