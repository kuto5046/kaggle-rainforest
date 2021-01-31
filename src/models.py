import os
import copy
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
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import accuracy_score
from src.dataset import SpectrogramDataset
import src.configuration as C
from src.metric import LWLRAP
import src.criterion as criterion
from src.conformer import ConformerBlock
import pytorch_lightning as pl
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from pytorch_lightning.metrics import F1

def calc_acc(pred, y):
    pred = torch.sigmoid(pred).detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    return accuracy_score(y, pred)

"""
############
 Audio tagging model
############
"""
class MeanTeacherLearner(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.student_model = model
        self.teacher_model = copy.deepcopy(model)
        for param in self.teacher_model.parameters():
            param.detach_()  # 勾配は計算しない
        
        self.output_key = config['model']['output_key']
        self.class_criterion = C.get_criterion(self.config)
        self.consistency_criterion = criterion.softmax_mse_loss
        self.f1 = F1(num_classes=24)

    def training_step(self, batch, batch_idx):
        (x1, x2), y = batch
        x2 = x2.set_grad_enabled(False)  # 勾配計算には使用しない
        batch_size = y.shape[0]

        p = random.random()
        do_mixup = True if p < self.config['mixup']['prob'] else False

        if self.config['mixup']['flag'] and do_mixup:
            x1, y, y_shuffle1, lam1 = mixup_data(x1, y, alpha=self.config['mixup']['alpha'])
            x2, y, y_shuffle2, lam2 = mixup_data(x2, y, alpha=self.config['mixup']['alpha'])

        student_output = self.student_model(x1)
        teacher_output = self.teacher_model(x2)
        student_logit = student_output[self.output_key]
        teacher_logit = teacher_output[self.output_key]

        teacher_logit = torch.tensor(teacher_logit.detach().data, requires_grad=False)  # 勾配計算しない
        consistency_weight = get_current_consistency_weight(self.current_epoch)
        consistency_loss = consistency_weight * self.consistency_criterion(student_logit, teacher_logit) / batch_size   
        # 通常のlossでlabelとのlossを計算(NOLABELのものへの対処は？)
        if self.config['mixup']['flag'] and do_mixup:
            class_loss = mixup_criterion(self.class_criterion, student_output, y, y_shuffle1, lam1, phase='train')
        else:
            class_loss = self.class_criterion(student_output, y, phase="train", nega_weight=consistency_weight)  # TODO yの-1はどう扱う？


        loss = class_loss + consistency_loss

        student_logit = student_logit[y.sum(axis=1) > 0]
        y = y[y.sum(axis=1) > 0]
        lwlrap = LWLRAP(student_logit, y)
        f1_score = self.f1(student_logit.sigmoid(), y)

        self.log(f'loss/train', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'LWLRAP/train', lwlrap, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'F1/train', f1_score, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    
    # batchのxはlist型
    def validation_step(self, batch, batch_idx):
        # xが複数の場合
        x_list, y = batch
        x = x_list.view(-1, x_list.shape[2], x_list.shape[3], x_list.shape[4])  # batch>1でも可
    
        output = self.student_model(x)
        class_loss = self.class_criterion(output, y, phase='valid')
        
        pred = output['logit']
        pred = C.split2one(pred, y)
        pred = pred[y.sum(axis=1) > 0]
        y = y[y.sum(axis=1) > 0]
        lwlrap = LWLRAP(pred, y)
        f1_score = self.f1(pred.sigmoid(), y)

        self.log(f'loss/val', class_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'LWLRAP/val', lwlrap, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'F1/val', f1_score, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return class_loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                    optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

        optimizer.step(closure=optimizer_closure)
        update_ema_variables(self.student_model, self.teacher_model, 0.999, self.global_step)


    def configure_optimizers(self):
        optimizer = C.get_optimizer(self.student_model, self.config)
        scheduler = C.get_scheduler(optimizer, self.config)
        return [optimizer], [scheduler]

# 最初は影響を小さくしておく
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 1.0 * sigmoid_rampup(epoch, 30)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


# Learner class(pytorch-lighting)
class Learner(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.model = model
        self.output_key = config["model"]["output_key"]
        self.criterion = C.get_criterion(self.config)
        self.f1 = F1(num_classes=24)

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        p = random.random()
        do_mixup = True if p < self.config['mixup']['prob'] else False

        if self.config['mixup']['flag'] and do_mixup:
            x, y, y_shuffle, lam = mixup_data(x, y, alpha=self.config['mixup']['alpha'])

        output = self.model(x)
        pred = output[self.output_key]
        if 'framewise' in self.output_key:
            pred, _ = pred.max(dim=1)
    
        if self.config['mixup']['flag'] and do_mixup:
            loss = mixup_criterion(self.criterion, output, y, y_shuffle, lam, phase='train')
        else:
            loss = self.criterion(output, y, phase="train")

        lwlrap = LWLRAP(pred, y)
        f1_score = self.f1(pred, y)

        self.log(f'loss/train', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'LWLRAP/train', lwlrap, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'F1/train', f1_score, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss
    
    # batchのxはlist型
    def validation_step(self, batch, batch_idx):
        # xが複数の場合
        x_list, y = batch
        x = x_list.view(-1, x_list.shape[2], x_list.shape[3], x_list.shape[4])  # batch>1でも可
    
        output = self.model(x)
        loss = self.criterion(output, y, phase='valid')
        pred = output[self.output_key]
        if 'framewise' in self.output_key:
            pred, _ = pred.max(dim=1)
        pred = C.split2one(pred, y)
        lwlrap = LWLRAP(pred, y)
        f1_score = self.f1(pred, y)
        self.log(f'loss/val', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'LWLRAP/val', lwlrap, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'F1/val', f1_score, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss


    def configure_optimizers(self):
        optimizer = C.get_optimizer(self.model, self.config)
        scheduler = C.get_scheduler(optimizer, self.config)
        return [optimizer], [scheduler]
    

# Learner class(pytorch-lighting)
class SAMLearner(Learner):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.config = config
        self.criterion = C.get_criterion(self.config)
        self.f1 = F1(num_classes=24)

    # DEFAULT
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                    optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

        optimizer.first_step(closure=optimizer_closure, zero_grad=True)
        optimizer.second_step(closure=optimizer_closure, zero_grad=True)


class Conformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_params = config['model']['params']


        # from https://arxiv.org/pdf/2007.03931.pdf
        self.convblock = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3,3)), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, kernel_size=(3,3)), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, kernel_size=(3,3)), nn.ReLU(), nn.MaxPool2d((1,2)),
            nn.Conv2d(64, 128, kernel_size=(3,3)), nn.ReLU(), nn.MaxPool2d((1,2)),
            nn.Conv2d(128, 128, kernel_size=(3,3)), nn.ReLU(), nn.MaxPool2d((1,2)),
            nn.Conv2d(128, 128, kernel_size=(3,3)), nn.ReLU(), nn.MaxPool2d((1,2)),
            nn.Conv2d(128, 128, kernel_size=(3,3)), nn.ReLU(), nn.MaxPool2d((1,2))  # modified by 1st team to fit output size
        )
        dim = 128
        """
        self.model = resnest50(pretrained=True)
        dim = self.model.fc.in_features
        layers = list(self.model.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        """
        self.conformerblock = ConformerBlock(dim=dim, **self.model_params)
        self.decoder = nn.Linear(dim, 24)

        self.init_weight()

    def init_weight(self):
        init_weights(self.convblock)
        init_weights(self.decoder)

    def forward(self, input):  # (batch, 3, 224, 496)
        x = self.convblock(input)  # (batch, 128, 44, 1)
        x = x.squeeze(3).permute((0, 2, 1))  # (batch, 44, 128)
        x = self.conformerblock(x)
        x = x.permute((0, 2, 1))  # (batch, 128, 44)
        x = F.adaptive_max_pool1d(x, 1).squeeze()  # (60, 128)
        # x = x.view(batch_size, -1)
        logit = self.decoder(x)  # (60, 24)
        output_dict = {'logit': logit}
        return output_dict


class ResNeStSED(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_params = config['model']['params']
        base_model = torch.hub.load("zhanghang1989/ResNeSt",
                                    model_params['base_model_name'],
                                    pretrained=model_params['pretrained'])

        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        in_features = base_model.fc.in_features
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, model_params['num_classes'], activation="sigmoid")

        self.init_weight()
        self.interpolate_ratio = 30  # Downsampled ratio

        self.model = nn.Sequential()

    def init_weight(self):
        init_layer(self.fc1)

class ResNet50(nn.Module):
    def __init__(self, config):
        super().__init__()
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


class ResNeSt50(nn.Module):
    def __init__(self, config):
        super().__init__()
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

        x = self.model(x)
        output_dict = {
            "logit": x 
        }
        return output_dict


"""
###########
  SED(Sound Event Detection) model
###########
"""
class PANNsCNN14AttSED(nn.Module):
    def __init__(self, config):
        super().__init__()

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

        # Input: (batch_size, data_length)


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


class ResNeStSED(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_params = config['model']['params']
        base_model = torch.hub.load("zhanghang1989/ResNeSt",
                                    model_params['base_model_name'],
                                    pretrained=model_params['pretrained'])

        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        in_features = base_model.fc.in_features
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, model_params['num_classes'], activation="sigmoid")

        self.init_weight()
        self.interpolate_ratio = 30  # Downsampled ratio

        self.model = nn.Sequential()

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
        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
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
    def __init__(self, config):
        super().__init__()

        model_params = config['model']['params'] 
        if model_params['pretrained']:
            self.base_model = EfficientNet.from_pretrained(model_params['base_model_name'])
        else:
            self.base_model = EfficientNet.from_name(model_params['base_model_name'])

        in_features = self.base_model._fc.in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, model_params['num_classes'], activation="sigmoid")

        self.init_weight()
        self.interpolate_ratio = 30  # Downsampled ratio

    def init_weight(self):
        init_layer(self.fc1)


    def forward(self, input):
        frames_num = input.size(3)

        # (batch_size, channels, freq, frames) ex->(120, 1408, 7, 12)
        x = self.base_model.extract_features(input)

        # (batch_size, channels, frames) ex->(120, 1408, 12)
        x = torch.mean(x, dim=2)

        # channel smoothing
        # channel次元上でpoolingを行う
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)  # torch.Size([120, 1408, 12]) -> torch.Size([120, 12, 1408])
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)  # torch.Size([120, 12, 1408]) -> torch.Size([120, 1408, 12])
        x = F.dropout(x, p=0.5, training=self.training)
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)  # claにsigmoidをかけない状態でclipwiseを計算
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)  # torch.Size([120, 12, 24])
        segmentwise_output = segmentwise_output.transpose(1, 2)  # torch.Size([120, 12, 24])

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)  # n_time次元上でをupsampling
        framewise_output = pad_framewise_output(framewise_output, frames_num)  # n_timesの最後の値で穴埋めしてframes_numに合わせる

        framewise_logit = interpolate(segmentwise_logit, self.interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "clipwise_output": clipwise_output,
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "segmentwise_logit": segmentwise_logit  
        }

        return output_dict


class ConformerSED(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_params = config['model']['params']
        self.interpolate_ratio = 9  # Downsampled ratio

        # from https://arxiv.org/pdf/2007.03931.pdf
        self.convblock = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3,3)), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, kernel_size=(3,3)), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, kernel_size=(3,3)), nn.ReLU(), nn.MaxPool2d((1,2)),
            nn.Conv2d(64, 128, kernel_size=(3,3)), nn.ReLU(), nn.MaxPool2d((1,2)),
            nn.Conv2d(128, 128, kernel_size=(3,3)), nn.ReLU(), nn.MaxPool2d((1,2)),
            nn.Conv2d(128, 128, kernel_size=(3,3)), nn.ReLU(), nn.MaxPool2d((1,2)),
            nn.Conv2d(128, 128, kernel_size=(3,3)), nn.ReLU(), nn.MaxPool2d((1,2))  # modified by 1st team to fit output size
        )
        self.conformerblock = ConformerBlock(dim=128, **self.model_params)
        self.linear = nn.Linear(128, 128, bias=True)
        self.att_block = AttBlockV2(128, 24, activation="sigmoid")


    def forward(self, input):
        batch_size = input.size(0)
        frames_num = input.size(3)

        x = self.convblock(input)
        x = x.squeeze(3).permute((0, 2, 1))  # (batch, channel, 44, 1) -> (batch, 44, channel)

        # conformer block was stacked 4 times
        x = self.conformerblock(x)
        x = self.conformerblock(x)
        x = self.conformerblock(x)
        x = self.conformerblock(x)
    
        # x = torch.mean(x, dim=1).unsqueeze(1)  # (batch, 44, ch)
        x = x.permute(0, 2, 1)  # (batch, ch, 44)

        # channel smoothing
        # channel次元上でpoolingを行う
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)  # torch.Size([batch, 1, 128]) -> torch.Size([batch, 128, 1])
        x = self.linear(x)
        x = F.relu_(x)
        x = x.transpose(1, 2)  # torch.Size([batch, 128, 1]) -> torch.Size([batch, 1, 128])
        x = F.dropout(x, p=0.5, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)  # claにsigmoidをかけない状態でclipwiseを計算
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)  # torch.Size([batch, 44, n_class])
        segmentwise_output = segmentwise_output.transpose(1, 2)  # torch.Size([batch, 44, n_class])

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)  # n_time次元上でをupsampling
        framewise_output = pad_framewise_output(framewise_output, frames_num)  # n_timesの最後の値で穴埋めしてframes_numに合わせる

        framewise_logit = interpolate(segmentwise_logit, self.interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "clipwise_output": clipwise_output,
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "segmentwise_logit": segmentwise_logit  
        }

        return output_dict

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

# n_timeの最後の値で穴埋めしてframe数になるようにする
def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. 
       The pad value is the same as the value of the last frame.
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
        """
        Args:
        x: (n_samples, n_in, n_time)  ex)torch.Size([120, 1408, 12])
        Outputs:
        x:(batch_size, classes_num) ex)torch.Size([120, 24])
        norm_att: batch_size, classes_num, n_time) ex)torch.Size([120, 24, 12])
        cla: batch_size, classes_num, n_time) ex)torch.Size([120, 24, 12])
        """
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)  # torch.Size([batch, n_class, 1]) クラス数に圧縮/valueを-1~1/n_timeの次元の総和=１に変換
        cla = self.nonlinear_transform(self.cla(x))  # self.cla()=self.att()/sigmoid変換
        x = torch.sum(norm_att * cla, dim=2)  # 要素同士の積 torch.Size([120, 24]): (batch, n_class)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


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


def mixup_criterion(criterion, pred, y_a, y_b, lam, phase='train'):
    return lam * criterion(pred, y_a, phase) + (1 - lam) * criterion(pred, y_b, phase)


def get_model(config: dict):
    model_name = config["model"]["name"]
    model_params = config["model"]["params"]
    if model_name == "ResNet50":
        model = ResNet50(config)
        learner = Learner(model, config)
    
    elif model_name == "ResNeSt50":
        model = ResNeSt50(config)
        learner = Learner(model, config)

    elif model_name == "ResNeSt50Sam":
        model = ResNeSt50(config)
        learner = SAMLearner(model, config)

    elif model_name == "Conformer":
        model = Conformer(config)
        learner = Learner(model, config)

    elif model_name == "PANNsCNN14Att":
        model = PANNsCNN14AttSED(config)  # TODO num_classes 527
        checkpoint = torch.load("pretrained/PANNsCNN14Att.pth")
        model.load_state_dict(checkpoint["model"])
        model.att_block = AttBlock(
            2048, model_params["n_classes"], activation="sigmoid")
        model.att_block.init_weights()
        init_layer(model.fc1)
        learner = Learner(model, config)
    elif model_name == "ResNeStSED":
        model = ResNeStSED(config)
        learner = Learner(model, config)
    elif model_name == "EfficientNetSED":
        model = EfficientNetSED(config)
        learner = MeanTeacherLearner(model, config)
    elif model_name == "ConformerSED":
        model = ConformerSED(config)
        learner = Learner(model, config)
    else:
        raise NotImplementedError

    
    return learner