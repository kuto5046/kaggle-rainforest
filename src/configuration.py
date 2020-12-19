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


def get_metadata(config: dict):
    data_config = config["data"]
    train = pd.read_csv(Path(data_config["root"]) / Path(data_config["train_df_path"])).reset_index(drop=True)
    train_audio_path = Path(data_config["root"]) / Path(data_config["train_audio_path"])

    # 破損しているファイルは除く
    # drop_ids = data_config["skip"]
    # drop_idx = train[train["recording_id"].isin(drop_ids)].index
    # train = train.drop(index=drop_idx).reset_index(drop=True)

    # drop_idx = train_tp[train_tp["recording_id"].isin(drop_ids)].index
    # train_tp = train_tp.drop(index=drop_idx).reset_index(drop=True)

    # drop_idx = train_fp[train_fp["recording_id"].isin(drop_ids)].index
    # train_fp = train_fp.drop(index=drop_idx).reset_index(drop=True)
    
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

    if phase == 'train':
        if dataset_config["name"] == "SpectrogramDataset":
            waveform_transforms = get_waveform_transforms(config)
            spectrogram_transforms = get_spectrogram_transforms(config)
            melspectrogram_parameters = dataset_config["params"]
            loader_config = config["loader"][phase]

            dataset = datasets.SpectrogramDataset(
                df,
                datadir=datadir,
                img_size=dataset_config["img_size"],
                waveform_transforms=waveform_transforms,
                spectrogram_transforms=spectrogram_transforms,
                melspectrogram_parameters=melspectrogram_parameters)
        else:
            raise NotImplementedError
        
    else:  # validとtest
        if dataset_config["name"] == "SpectrogramDataset":
            waveform_transforms = get_waveform_transforms(config)
            spectrogram_transforms = get_spectrogram_transforms(config)
            melspectrogram_parameters = dataset_config["params"]
            loader_config = config["loader"][phase]

            dataset = datasets.SpectrogramTestDataset(
                df,
                datadir=datadir,
                img_size=dataset_config["img_size"],
                waveform_transforms=waveform_transforms,
                spectrogram_transforms=spectrogram_transforms,
                melspectrogram_parameters=melspectrogram_parameters)
        else:
            raise NotImplementedError
    loader = data.DataLoader(dataset, **loader_config)
    return loader

