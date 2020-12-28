import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import sklearn.model_selection as sms

import src.dataset as datasets

from pathlib import Path
from sklearn.metrics import label_ranking_loss
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

"""
def lr_loss(output, target):
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    loss = label_ranking_loss(target, output)

    return loss
"""

"""
def LwlrapLoss(output, target):
    # Ranks of the predictions
    ranked_classes = torch.argsort(output, dim=-1, descending=True)
    # i, j corresponds to rank of prediction in row i
    class_ranks = torch.zeros_like(ranked_classes)
    for i in range(ranked_classes.size(0)):
        for j in range(ranked_classes.size(1)):
            class_ranks[i, ranked_classes[i][j]] = j + 1
    # Mask out to only use the ranks of relevant GT labels
    ground_truth_ranks = class_ranks * target + (1e6) * (1 - target)
    # All the GT ranks are in front now
    sorted_ground_truth_ranks, _ = torch.sort(ground_truth_ranks, dim=-1, descending=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pos_matrix = torch.tensor(np.array([i+1 for i in range(target.size(-1))])).unsqueeze(0).to(device)

    mask_matrix, _ = torch.sort(target, dim=-1, descending=True)
    true_ranks = pos_matrix * mask_matrix
    pred_ranks = sorted_ground_truth_ranks * mask_matrix
    loss = nn.functional.mse_loss(pred_ranks, true_ranks, size_average=None, reduce=None, reduction='mean')
    loss.requires_grad = True
    return loss
"""
# refered following repo
# https://github.com/ex4sperans/freesound-classification/blob/71b9920ce0ae376aa7f1a3a2943f0f92f4820813/networks/losses.py
def lsep_loss_stable(input, target, average=True):

    n = input.size(0)

    differences = input.unsqueeze(1) - input.unsqueeze(2)
    where_lower = (target.unsqueeze(1) < target.unsqueeze(2)).float()

    differences = differences.view(n, -1)
    where_lower = where_lower.view(n, -1)

    max_difference, index = torch.max(differences, dim=1, keepdim=True)
    differences = differences - max_difference
    exps = differences.exp() * where_lower

    lsep = max_difference + torch.log(torch.exp(-max_difference) + exps.sum(-1))

    if average:
        return lsep.mean()
    else:
        return lsep


def lsep_loss(input, target, average=True):

    differences = input.unsqueeze(1) - input.unsqueeze(2)
    where_different = (target.unsqueeze(1) < target.unsqueeze(2)).float()

    exps = differences.exp() * where_different
    lsep = torch.log(1 + exps.sum(2).sum(1))

    if average:
        return lsep.mean()
    else:
        return lsep


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
        criterion = globals().get(loss_name)
        # if criterion_cls is not None:
        #     criterion = criterion_cls(**loss_params)
        # else:
        #     raise NotImplementedError

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
                waveform_transforms=get_waveform_transforms(config, phase),
                spectrogram_transforms=get_spectrogram_transforms(config, phase),
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
                waveform_transforms=get_waveform_transforms(config, phase),
                spectrogram_transforms=get_spectrogram_transforms(config, phase),
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
                waveform_transforms=get_waveform_transforms(config, phase),
                spectrogram_transforms=get_spectrogram_transforms(config, phase),
                melspectrogram_parameters=dataset_config["params"])
        else:
            raise NotImplementedError

    loader_config = config["loader"][phase]
    loader = data.DataLoader(dataset, **loader_config)
    return loader

