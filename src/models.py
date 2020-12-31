import os
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
        pred = self.forward(x)
        criterion = C.get_criterion(self.config)
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

    def forward(self, x, mixup_lambda=None):
        """
        # spec aug
        if self.training:
            x = self.spec_augmenter(x)
        """
        """
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        """
        x = self.model(x)
        return x



class ResNeSt50SamLearner(Learner):
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

    def forward(self, x, mixup_lambda=None):
        """
        # spec aug
        if self.training:
            x = self.spec_augmenter(x)
        """
        """
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        """
        x = self.model(x)
        return x

    # def backward(self, loss, optimizer, optimizer_idx):
    #     loss.backward()

    # DEFAULT
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                    optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

        optimizer.first_step(closure=optimizer_closure, zero_grad=True)
        optimizer.second_step(closure=optimizer_closure, zero_grad=True)


def get_model(config, fold):
    model_name = config["model"]["name"]
    if model_name == "ResNet50":
        model = ResNet50Learner(config, fold)
        return model
    elif model_name == "ResNeSt50":
        model = ResNeSt50Learner(config, fold)
        return model
    elif model_name == "ResNeSt50Sam":
        model = ResNeSt50SamLearner(config, fold)
        return model
    else:
        raise NotImplementedError


def do_mixup(x: torch.Tensor, mixup_lambda: torch.Tensor):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes
    (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)
    Returns:
      out: (batch_size, ...)
    """
    out = (x[0::2].transpose(0, -1) * mixup_lambda[0::2] +
           x[1::2].transpose(0, -1) * mixup_lambda[1::2]).transpose(0, -1)
    return out


class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return torch.from_numpy(np.array(mixup_lambdas, dtype=np.float32))


def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output


def mixup(x, y, num_classes=24, gamma=0.2, smooth_eps=0.1):
    if gamma == 0 and smooth_eps == 0:
        return x, y
    m = Beta(torch.tensor([gamma]), torch.tensor([gamma]))
    lambdas = m.sample([x.size(0), 1, 1]).to(x)
    my = onehot(y, num_classes).to(x)
    true_class, false_class = 1. - smooth_eps * num_classes / (num_classes - 1), smooth_eps / (num_classes - 1)
    my = my * true_class + torch.ones_like(my) * false_class
    perm = torch.randperm(x.size(0))
    x2 = x[perm]
    y2 = my[perm]
    return x * (1 - lambdas) + x2 * lambdas, my * (1 - lambdas) + y2 * lambdas


class Mixup(torch.nn.Module):
    def __init__(self, num_classes=24, gamma=0.2, smooth_eps=0.1):
        super(Mixup, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smooth_eps = smooth_eps

    def forward(self, input, target):
        return mixup(input, target, self.num_classes, self.gamma, self.smooth_eps)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_h = np.int(H * cut_rat)

    # uniform
    cy = np.random.randint(int(H / 5), int(4 * H / 5))
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return 0, bby1, W, bby2


def cutmix_or_mixup(data, targets, is_cutmix=False, use_all=False):
    # cutmix if is_cutmix else mixup
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets0 = targets[:, 0][indices]
    shuffled_targets1 = targets[:, 1][indices]
    shuffled_targets2 = targets[:, 2][indices]

    lam = np.random.uniform(0, 1.0)
    if is_cutmix:
        bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
        data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[indices, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    else:
        data = data * lam + shuffled_data * (1 - lam)

    out = [targets[:, 0], shuffled_targets0,
           targets[:, 1], shuffled_targets1,
           targets[:, 2], shuffled_targets2]
    if use_all:
        shuffled_targets3 = targets[:, 3][indices]
        out.append(targets[:, 3])
        out.append(shuffled_targets3)
    return data, out, lam