import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import sklearn.model_selection as sms

import src.dataset as datasets

from pathlib import Path
from src.transforms import (get_waveform_transforms,
                            get_spectrogram_transforms)



def get_device(device: str):
    return torch.device(device)


def get_optimizer(model: nn.Module, config: dict):
    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config.get("name")

    return optim.__getattribute__(optimizer_name)(model.parameters(),
                                                  **optimizer_config["params"])


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

    # TODO 要変更
    pos_weight = torch.ones(loss_config["num_classes"]).to(config["globals"]["device"])
    loss_params["pos_weight"] = pos_weight * loss_config["num_classes"]

    if hasattr(nn, loss_name):
        criterion = nn.__getattribute__(loss_name)(**loss_params)
    else:
        criterion_cls = globals().get(loss_name)
        if criterion_cls is not None:
            criterion = criterion_cls(**loss_params)
        else:
            raise NotImplementedError

    return criterion


def get_split(config: dict):
    split_config = config["split"]
    name = split_config["name"]

    return sms.__getattribute__(name)(**split_config["params"])

"""
# wav data
def get_metadata(config: dict):
    data_config = config["data"]
    train = pd.read_csv(Path(data_config["root"]) / Path(data_config["train_df_path"])).reset_index(drop=True)
    train_audio_path = Path(data_config["root"]) / Path(data_config["train_audio_path"])

    if data_config['use_train_data'] == ['tp']:
        # train = train.iloc[:500,:]  # メモリ対策
        return train[train['data_type']=='tp'], train_audio_path
    elif data_config['use_train_data'] == ['fp']:
        return train[train['data_type']=='fp'], train_audio_path
    elif data_config['use_train_data'] == ['tp', 'fp']:
        return train, train_audio_path
    else:
        print("exception error")
"""

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
        # train = train.iloc[:500,:]  # メモリ対策
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
                waveform_transforms=get_waveform_transforms(config),
                spectrogram_transforms=get_spectrogram_transforms(config),
                melspectrogram_parameters=dataset_config["params"])
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
                waveform_transforms=get_waveform_transforms(config),
                spectrogram_transforms=get_spectrogram_transforms(config),
                melspectrogram_parameters=dataset_config["params"])
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
                waveform_transforms=get_waveform_transforms(config),
                spectrogram_transforms=get_spectrogram_transforms(config),
                melspectrogram_parameters=dataset_config["params"])
        else:
            raise NotImplementedError

    loader_config = config["loader"][phase]
    loader = data.DataLoader(dataset, **loader_config)
    return loader

