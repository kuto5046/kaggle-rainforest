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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.criterion = C.get_criterion(self.config)

    
    def training_step(self, batch, batch_idx):
        x, y = batch

        p = random.random()
        do_mixup = True if p < self.config['mixup']['prob'] else False

        if self.config['mixup']['flag'] and do_mixup:
            x, y, y_shuffle, lam = mixup_data(x, y, alpha=self.config['mixup']['alpha'])

        pred = self.forward(x)

        if self.config['mixup']['flag'] and do_mixup:
            loss = mixup_criterion(self.criterion, pred, y, y_shuffle, lam)
        else:
            loss = self.criterion(pred, y)

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

        # Spec augmenter
        # self.spec_augmenter = SpecAugmentation(
        #     time_drop_width=64,
        #     time_stripes_num=2,
        #     freq_drop_width=8,
        #     freq_stripes_num=2)

    def forward(self, x):
        """
        # spec aug
        if self.training:
            x = self.spec_augmenter(x)
        """
        x = self.model(x)
        return x


class ResNeSt50SamLearner(Learner):
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

        # Spec augmenter
        # self.spec_augmenter = SpecAugmentation(
        #     time_drop_width=64,
        #     time_stripes_num=2,
        #     freq_drop_width=8,
        #     freq_stripes_num=2)

    def forward(self, x):
        """
        # spec aug
        if self.training:
            x = self.spec_augmenter(x)
        """

        x = self.model(x)
        return x


    # DEFAULT
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                    optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

        optimizer.first_step(closure=optimizer_closure, zero_grad=True)
        optimizer.second_step(closure=optimizer_closure, zero_grad=True)


class PANNsCNN14AttLearner(Learner):
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


class PANNsCNN14AttLearner(Learner):
    def __init__(self, config):
        super().__init__(config)

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.interpolate_ratio = 32  # Downsampled ratio
        model_params = config['model']['params']

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=model_params['n_fft'],
            hop_length=model_params['hop_length'],
            win_length=model_params['win_length'],
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=model_params['sr'],
            n_fft=model_params['n_fft'],
            n_mels=model_params['n_mels'],
            fmin=model_params['fmin'],
            fmax=model_params['fmax'],
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(model_params['n_mels'])

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.att_block = AttBlock(2048, model_params['num_classes'], activation='sigmoid')

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        # t1 = time.time()
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        # if self.training and mixup_lambda is not None:
        #     x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        output_dict = {
            'framewise_output': framewise_output,
            "logit": logit,
            'clipwise_output': clipwise_output
        }

        return output_dict



def get_model(config):
    model_name = config["model"]["name"]
    if model_name == "ResNet50":
        model = ResNet50Learner(config)
        return model
    elif model_name == "ResNeSt50":
        model = ResNeSt50Learner(config)
        return model
    elif model_name == "ResNeSt50Sam":
        model = ResNeSt50SamLearner(config)
        return model
    elif model_name == "PANNsCNN14Att":
        model = PANNsCNN14AttLearner(config)
    else:
        raise NotImplementedError

"""
############
   mixup
############
"""

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


"""
##########
    SED
##########
"""
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()

def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class AttBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear",
                 temperature=1.0):
        super().__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.bn_att = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


""""
############
 SED other archetecture
############
"""

class ResNestSED(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False,
                 num_classes=264):
        super().__init__()
        self.interpolate_ratio = 30  # Downsampled ratio
        base_model = torch.hub.load("zhanghang1989/ResNeSt",
                                    base_model_name,
                                    pretrained=pretrained)
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.fc.in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)

    def forward(self, input):
        frames_num = input.size(3)

        # (batch_size, channels, freq, frames)
        x = self.encoder(input)

        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, self.interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output
        }

        return output_dict


class EfficientNetSED(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False,
                 num_classes=264):
        super().__init__()
        self.interpolate_ratio = 30  # Downsampled ratio
        if pretrained:
            self.base_model = EfficientNet.from_pretrained(base_model_name)
        else:
            self.base_model = EfficientNet.from_name(base_model_name)

        in_features = self.base_model._fc.in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)

    def forward(self, input):
        frames_num = input.size(3)

        # (batch_size, channels, freq, frames)
        x = self.base_model.extract_features(input)

        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, self.interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output
        }

        return output_dict

"""
def get_model(config: dict):
    model_config = config["model"]
    model_name = model_config["name"]
    model_params = model_config["params"]

    if model_name == "PANNsCNN14Att":
        if model_params["pretrained"]:
            model = PANNsCNN14Att(  # type: ignore
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527)
            checkpoint = torch.load("pretrained/PANNsCNN14Att.pth")
            model.load_state_dict(checkpoint["model"])

            model.att_block = AttBlock(
                2048, model_params["n_classes"], activation="sigmoid")
            model.att_block.init_weights()
            init_layer(model.fc1)
        else:
            model = PANNsCNN14Att(  # type: ignore
                sample_rate=model_params["sample_rate"],
                window_size=model_params["window_size"],
                hop_size=model_params["hop_size"],
                mel_bins=model_params["mel_bins"],
                fmin=model_params["fmin"],
                fmax=model_params["fmax"],
                classes_num=model_params["n_classes"])
        return model
    elif model_name == "ResNestSED":
        model = ResNestSED(  # type: ignore
            **model_params)
        return model
    elif model_name == "EfficientNetSED":
        model = EfficientNetSED(  # type: ignore
            **model_params)
        return model
    else:
        raise NotImplementedError


def get_model_for_inference(config: dict, weights_dir: str):
    model_config = config["model"]
    model_name = model_config["name"]
    model_params = model_config["params"]

    if model_name == "PANNsCNN14Att":
        if model_params["pretrained"]:
            params = {
                "sample_rate": 32000,
                "window_size": 1024,
                "hop_size": 320,
                "mel_bins": 64,
                "fmin": 50,
                "fmax": 14000,
                "classes_num": model_params["n_classes"]
            }
            model = PANNsCNN14Att(**params)  # type: ignore
        else:
            model = PANNsCNN14Att(  # type: ignore
                sample_rate=model_params["sample_rate"],
                window_size=model_params["window_size"],
                hop_size=model_params["hop_size"],
                mel_bins=model_params["mel_bins"],
                fmin=model_params["fmin"],
                fmax=model_params["fmax"],
                classes_num=model_params["n_classes"])
    elif model_name == "ResNestSED":
        model = ResNestSED(  # type: ignore
            base_model_name=model_params["base_model_name"],
            pretrained=False,
            num_classes=model_params["num_classes"])
    else:
        raise NotImplementedError

    if not torch.cuda.is_available():
        weights = torch.load(weights_dir, map_location="cpu")
    else:
        weights = torch.load(weights_dir)
    model.load_state_dict(weights["model_state_dict"])
    return model


"""