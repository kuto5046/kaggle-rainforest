import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import torchvision
from resnest.torch import resnest50
from sklearn.metrics import accuracy_score
from src.dataset import SpectrogramDataset
import src.configuration as C
from src.metric import LWLRAP
import pytorch_lightning as pl


# Learner class(pytorch-lighting)
class Learner(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        return None
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        criterion = C.get_criterion(self.config)
        loss = criterion(pred, y)
        lwlrap = LWLRAP(pred, y)
        # acc = accuracy_score(y, pred)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_LWLRAP', lwlrap, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # #self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
        # acc = accuracy_score(y, pred)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_LWLRAP', lwlrap, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)  
        return loss

    def configure_optimizers(self):
        optimizer = C.get_optimizer(self.model, self.config)
        scheduler = C.get_scheduler(optimizer, self.config)
        return [optimizer], [scheduler]


class ResNet50Learner(Learner):
    def __init__(self, config):
        super().__init__(config)
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
    def __init__(self, config):
        super().__init__(config)
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

    def forward(self, x):
        return self.model(x)



def get_model(config):
    model_name = config["model"]["name"]
    if model_name == "ResNet50":
        model = ResNet50Learner(config)
        return model
    elif model_name == "ResNeSt50":
        model = ResNeSt50Learner(config)
        return model
    else:
        raise NotImplementedError

