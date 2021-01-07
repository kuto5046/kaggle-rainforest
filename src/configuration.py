import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import sklearn.model_selection as sms
from src.sam import SAM
import src.dataset as datasets

from pathlib import Path
from sklearn.metrics import label_ranking_loss
from src.criterion import LSEPLoss, LSEPStableLoss, ImprovedPANNsLoss
from src.transforms import (get_waveform_transforms,
                            get_spectrogram_transforms)

def get_device(device: str):
    return torch.device(device)


def get_optimizer(model: nn.Module, config: dict):
    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config.get("name")
    base_optimizer_name = optimizer_config.get("base_name")
    optimizer_params = optimizer_config['params']

    if hasattr(optim, optimizer_name):
        optimizer = optim.__getattribute__(optimizer_name)(model.parameters(), **optimizer_params)
        return optimizer
    else:
        base_optimizer = optim.__getattribute__(base_optimizer_name)
        optimizer = globals().get(optimizer_name)(
            model.parameters(), 
            base_optimizer,
            **optimizer_config["params"])
        return  optimizer

def get_scheduler(optimizer, config: dict):
    scheduler_config = config["scheduler"]
    scheduler_name = scheduler_config.get("name")

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(
            optimizer, **scheduler_config["params"])


def get_criterion(config: dict):
    loss_config = config["loss"]
    loss_name = loss_config["name"]
    loss_params = {} if loss_config.get("params") is None else loss_config.get("params")

    if ('BCE' in loss_name):
        pos_weight = torch.ones(loss_config["num_classes"]).to(config["globals"]["device"])
        loss_params["pos_weight"] = pos_weight * loss_config["num_classes"]

    if hasattr(nn, loss_name):
        criterion = nn.__getattribute__(loss_name)(**loss_params)
    else:
        criterion = globals().get(loss_name)(**loss_params)

    return criterion


def get_split(config: dict):
    split_config = config["split"]
    name = split_config["name"]

    return sms.__getattribute__(name)(**split_config["params"])


# flac data
def get_metadata(config: dict):
    data_config = config["data"]
    train_audio_path = Path(data_config["root"]) / Path(data_config["train_audio_path"])
    train_tp = pd.read_csv(Path(data_config["root"]) / Path(data_config["train_tp_df_path"])).reset_index(drop=True)
    train_fp = pd.read_csv(Path(data_config["root"]) / Path(data_config["train_fp_df_path"])).reset_index(drop=True)
    train_tp["data_type"] = "tp"
    train_fp["data_type"] = "fp"
    train = pd.concat([train_tp, train_fp])
    if data_config['use_train_data'] == ['tp']:
        return train[train['data_type']=='tp'], train_audio_path
    elif data_config['use_train_data'] == ['fp']:
        return train[train['data_type']=='fp'], train_audio_path
    elif data_config['use_train_data'] == ['tp', 'fp']:
        return train, train_audio_path
    else:
        print("exception error")


def get_test_metadata(config: dict):
    data_config = config["data"]
    sub = pd.read_csv(Path(data_config["root"]) / Path(data_config["sub_df_path"])).reset_index(drop=True)
    test_audio_path = Path(data_config["root"]) / Path(data_config["test_audio_path"])

    return sub, test_audio_path


def get_loader(df: pd.DataFrame,
               datadir: Path,
               config: dict,
               phase: str):

    dataset_config = config["dataset"]

    # train
    if phase == 'train':
        if dataset_config["name"] == "SpectrogramDataset":
            dataset = datasets.SpectrogramDataset(
                df,
                datadir=datadir,
                height=dataset_config["height"],
                width=dataset_config["width"],
                period=dataset_config['period'],
                strong_label_prob=dataset_config['strong_label_prob'],
                waveform_transforms=get_waveform_transforms(config, phase),
                spectrogram_transforms=get_spectrogram_transforms(config, phase),
                melspectrogram_parameters=dataset_config["params"]['melspec'],
                pcen_parameters=dataset_config['params']['pcen'])
        else:
            raise NotImplementedError
    # valid    
    elif phase == 'valid':
        if dataset_config["name"] == "SpectrogramDataset":
            dataset = datasets.SpectrogramValDataset(
                df,
                datadir=datadir,
                height=dataset_config["height"],
                width=dataset_config["width"],
                period=dataset_config['period'],
                shift_time=dataset_config['shift_time'],
                waveform_transforms=get_waveform_transforms(config, phase),
                spectrogram_transforms=get_spectrogram_transforms(config, phase),
                melspectrogram_parameters=dataset_config["params"]['melspec'],
                pcen_parameters=dataset_config['params']['pcen'])
        else:
            raise NotImplementedError
    # test
    else:
        if dataset_config["name"] == "SpectrogramDataset":
            dataset = datasets.SpectrogramTestDataset(
                df,
                datadir=datadir,
                height=dataset_config["height"],
                width=dataset_config["width"],
                period=dataset_config['period'],
                shift_time=dataset_config['shift_time'],
                waveform_transforms=get_waveform_transforms(config, phase),
                spectrogram_transforms=get_spectrogram_transforms(config, phase),
                melspectrogram_parameters=dataset_config["params"]['melspec'],
                pcen_parameters=dataset_config['params']['pcen'])
        else:
            raise NotImplementedError

    loader_config = config["loader"][phase]
    loader = data.DataLoader(dataset, **loader_config, worker_init_fn=worker_init_fn)
    return loader

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def split2one(input, target):
    """
    validは60sの音声をn分割されバッチとしてモデルに入力される
    そこで出力後に分割されたデータを１つのデータに変換する必要がある
    クラスごとのmaxを出力とする
    """
    input_ = input.view(target.shape[0], -1, target.shape[1])  # y.shape[1]==num_classes
    input_ = torch.max(input_, dim=1)[0]  # 1次元目(分割sしたやつ)で各クラスの最大を取得
    return input_