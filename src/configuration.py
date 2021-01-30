import pandas as pd
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.sampler import Sampler, RandomSampler
import sklearn.model_selection as sms
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from src.sam import SAM
import src.dataset as datasets

from pathlib import Path
from sklearn.metrics import label_ranking_loss
from src.criterion import LSEPLoss, LSEPStableLoss, ImprovedPANNsLoss, BCEWithLogitsLoss, FocalLoss
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

    # if hasattr(nn, loss_name):
    #     criterion = nn.__getattribute__(loss_name)(**loss_params)
    # else:
    criterion = globals().get(loss_name)(**loss_params)

    return criterion


def get_split(config: dict):
    split_config = config["split"]
    split_name = split_config["name"]
    if hasattr(sms, split_name):
        return sms.__getattribute__(split_name)(**split_config["params"])
    else:
        return globals().get(split_name)(**split_config["params"])


# flac data
def get_metadata(config: dict):
    data_config = config["data"]
    train_audio_path = Path(data_config["root"]) / Path(data_config["train_audio_path"])
    train_tp = pd.read_csv(Path(data_config["root"]) / Path(data_config["train_tp_df_path"])).reset_index(drop=True)
    train_fp = pd.read_csv(Path(data_config["root"]) / Path(data_config["train_fp_df_path"])).reset_index(drop=True)
    # train_tp_frame_ps = pd.read_csv(Path(data_config["root"]) / Path(data_config["train_tp_frame_ps_df_path"])).reset_index(drop=True)

    train_tp["data_type"] = "tp"
    train_fp["data_type"] = "fp"
    # train_fp = train_fp.sample(1000)

    UNLABEL = -1
    train_fp["species_id"] = UNLABEL
    # train_tp_frame_ps["data_type"] = 'tp'
    # train_fp['species_id'] = 0  # ダミーデータ あとで上書きする

    # train = pd.concat([train_tp, train_fp[['recording_id', 'species_id', 'data_type', 't_min', 't_max']]])
    train = pd.concat([train_tp, train_fp])
    df = train[train['data_type'].isin(data_config['use_train_data'])].reset_index(drop=True)

    return df, train_audio_path


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
    loader_config = config["loader"][phase]

    if phase == "train":
        dataset = datasets.SpectrogramDataset(
            df,
            phase,
            datadir=datadir,
            height=dataset_config["height"],
            width=dataset_config["width"],
            period=dataset_config['period'],
            shift_time=dataset_config['shift_time'],
            strong_label_prob=dataset_config['strong_label_prob'],
            waveform_transforms=get_waveform_transforms(config, phase),
            spectrogram_transforms=get_spectrogram_transforms(config, phase),
            melspectrogram_parameters=dataset_config["params"]['melspec'],
            pcen_parameters=dataset_config['params']['pcen'])

        labeled_idxs = df[df["data_type"]=="tp"].index
        unlabeled_idxs = df[df["data_type"]=="fp"].index
        batch_sampler = TwoStreamBatchSampler(unlabeled_idxs, labeled_idxs, **loader_config)  # tp:fp=1:4
        loader = data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=10, worker_init_fn=worker_init_fn)

    else:
        dataset = datasets.SpectrogramDataset(
            df,
            phase,
            datadir=datadir,
            height=dataset_config["height"],
            width=dataset_config["width"],
            period=dataset_config['period'],
            shift_time=dataset_config['shift_time'],
            strong_label_prob=dataset_config['strong_label_prob'],
            waveform_transforms=get_waveform_transforms(config, phase),
            spectrogram_transforms=get_spectrogram_transforms(config, phase),
            melspectrogram_parameters=dataset_config["params"]['melspec'],
            pcen_parameters=dataset_config['params']['pcen'])

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



class TwoStreamBatchSampler(RandomSampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):  # secondalry == labeled
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size



def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
