import os
import torch
import subprocess
import traceback
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from tqdm import tqdm
# n.pyimport hydra
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.metric import LWLRAP
from src.models import get_model
import src.configuration as C
import src.utils as utils
from src.metric import LWLRAP
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.metrics.classification import F1, Recall, Precision
os.environ['NUMEXPR_MAX_THREADS'] = '24'


def make_oof(model, val_df, datadir, config, fold, output_dir):
    df = val_df[~val_df["recording_id"].duplicated()].reset_index(drop=True)
    loader = C.get_loader(df, datadir, config, phase="test")
    output_key = config['model']['output_key']
    all_oof_df = pd.DataFrame()
    os.makedirs("./oof", exist_ok=True)
    with torch.no_grad():
        # xは複数のlist
        for x_list, recording_id in tqdm(loader):
            oof_df = pd.DataFrame()
            oof_df["patch"] = [0,1,2,3,4,5,6,7]
            oof_df["recording_id"] = recording_id[0]
            columns = [f"s{i}" for i in range(24)]
            for col in columns:
                oof_df[col] = 0

            batch_size = x_list.shape[0]
            x = x_list.view(-1, x_list.shape[2], x_list.shape[3], x_list.shape[4])  # batch>1でも可
            x = x.to("cuda")
 
            output = model.model(x, None, False)
            output = output[output_key]
            output = output.view(batch_size, -1, 24).cpu().sigmoid()[0]  # 24=num_classes

            oof_df.loc[:, 's0':] = output
            all_oof_df = pd.concat([all_oof_df, oof_df])
    all_oof_df.to_csv(output_dir / f"/fold{fold}_oof.csv", index=False)
    return all_oof_df


def make_test(model, test_loader, datadir, config, fold, output_dir):

    output_key = config['model']['output_key']
    all_test_df = pd.DataFrame()
    os.makedirs("./oof", exist_ok=True)
    with torch.no_grad():
        # xは複数のlist
        for x_list, recording_id in tqdm(test_loader):
            test_df = pd.DataFrame()
            test_df["patch"] = [0,1,2,3,4,5,6,7]
            test_df["recording_id"] = recording_id[0]
            columns = [f"s{i}" for i in range(24)]
            for col in columns:
                test_df[col] = 0

            batch_size = x_list.shape[0]
            x = x_list.view(-1, x_list.shape[2], x_list.shape[3], x_list.shape[4])  # batch>1でも可
            x = x.to("cuda")
 
            output = model.model(x)
            output = output[output_key]
            output = output.view(batch_size, -1, 24).cpu().sigmoid()[0]  # 24=num_classes

            test_df.loc[:, 's0':] = output
            all_test_df = pd.concat([all_test_df, test_df])
    all_test_df.to_csv(output_dir / f"./oof/fold{fold}_test.csv", index=False)



def main():
    warnings.filterwarnings('ignore')

    # config
    config_filename = 'EfficientNetSED004_oof.yaml'
    config = utils.load_config(f"./configs/{config_filename}")
    global_config = config['globals']
    timestamp = utils.get_timestamp(config)
    output_dir = Path(global_config['output_dir']) / timestamp
    output_dir.mkdir(exist_ok=True, parents=True)
    # utils.send_slack_message_notification(f'[START] timestamp: {timestamp}')

    # utils config
    logger = utils.get_logger(output_dir/ "output.log")
    utils.set_seed(global_config['seed'])
    device = C.get_device(global_config["device"])

    # data
    df, datadir, tp_fnames, tp_labels, fp_fnames, fp_labels = C.get_metadata(config)
    sub_df, test_datadir = C.get_test_metadata(config)
    test_loader = C.get_loader(sub_df, test_datadir, config, phase="test")
    splitter1 = C.get_split(config)
    splitter2 = C.get_split(config)

    # Make CV
    tp_cv = [(np.array(tp_fnames)[train_index], np.array(tp_fnames)[valid_index]) for train_index, valid_index in splitter1.split(tp_fnames, tp_labels)]
    fp_cv = [(np.array(fp_fnames)[train_index], np.array(fp_fnames)[valid_index]) for train_index, valid_index in splitter2.split(fp_fnames, fp_labels)]

    all_oof = pd.DataFrame()
    for fold in range(5):
        # 指定したfoldのみループを回す
        if fold not in global_config['folds']:
            continue

        # tensorboard logger
        loggers = []
        model_name = config["model"]['name']
        tb_logger = TensorBoardLogger(save_dir=output_dir, name=model_name)
        loggers.append(tb_logger)

        logger.info('=' * 20)
        logger.info(f'Fold {fold}')
        logger.info('=' * 20)

        # dataloader
        tp_train, tp_valid = tp_cv[fold]  # tp data(stage1に合わせてleakしないように)
        fp_train, fp_valid = fp_cv[fold]  # fp data(positive labelで均等に分割)
        train_fname = np.hstack([tp_train, fp_train])
        valid_fname = np.hstack([tp_valid, fp_valid])

        trn_df = df[df['recording_id'].isin(train_fname)].reset_index(drop=True)
        val_df = df[df['recording_id'].isin(valid_fname)].reset_index(drop=True)

        # model
        model = get_model(config)

        """
        ##############
        predict part
        ##############
        """
        # load model
        try:
            ckpt = torch.load(output_dir / f'{model_name}-{fold}-v0.ckpt')
        except:
            ckpt = torch.load(output_dir / f'{model_name}-{fold}.ckpt')
        model.load_state_dict(ckpt['state_dict'])
        model.eval().to(device)
        

        # oof
        fold_oof = make_oof(model, val_df, datadir, config, fold, output_dir)
        all_oof = pd.concat([all_oof, fold_oof])
        make_test(model, test_loader, test_datadir, config, fold, output_dir)

    all_oof = all_oof.reset_index(drop=True)
    all_oof.to_csv(output_dir / "all_oof.csv", index=False)


if __name__ == '__main__':
    with utils.timer('Total time'):
        main()

