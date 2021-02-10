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


def make_oof(model, df, datadir, config, fold):
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
 
            output = model.model(x)
            output = output[output_key]
            output = output.view(batch_size, -1, 24).cpu().sigmoid()[0]  # 24=num_classes

            oof_df.loc[:, 's0':] = output
            all_oof_df = pd.concat([all_oof_df, oof_df])
    all_oof_df.to_csv(f"./oof/fold{fold}_oof.csv", index=False)


def make_test(model, test_loader, datadir, config, fold):

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
    all_test_df.to_csv(f"./oof/fold{fold}_test.csv", index=False)



def make_fp(model, df, datadir, config, fold):
    loader = C.get_loader(df, datadir, config, phase="test")
    output_key = config['model']['output_key']
    all_fp_df = pd.DataFrame()
    os.makedirs("./oof", exist_ok=True)
    with torch.no_grad():
        # xは複数のlist
        for x_list, recording_id in tqdm(loader):
            fp_df = pd.DataFrame()
            fp_df["patch"] = [0,1,2,3,4,5,6,7]
            fp_df["recording_id"] = recording_id[0]
            columns = [f"s{i}" for i in range(24)]
            for col in columns:
                fp_df[col] = 0

            batch_size = x_list.shape[0]
            x = x_list.view(-1, x_list.shape[2], x_list.shape[3], x_list.shape[4])  # batch>1でも可
            x = x.to("cuda")
 
            output = model.model(x)
            output = output[output_key]
            output = output.view(batch_size, -1, 24).cpu().sigmoid()[0]  # 24=num_classes

            fp_df.loc[:, 's0':] = output
            all_fp_df = pd.concat([all_fp_df, fp_df])
    all_fp_df.to_csv(f"./oof/fold{fold}_fp.csv", index=False)

def main():
    warnings.filterwarnings('ignore')

    # config
    config_filename = 'EfficientNetSED002.yaml'
    config = utils.load_config(f"configs/{config_filename}")
    global_config = config['globals']
    hash_value = utils.get_hash(config)  # get git hash value(short ver.)
    timestamp = utils.get_timestamp(config)
    output_dir = Path(global_config['output_dir']) / timestamp
    output_dir.mkdir(exist_ok=True, parents=True)
    # utils.send_slack_message_notification(f'[START] timestamp: {timestamp}')

    # utils config
    logger = utils.get_logger(output_dir/ "output.log")
    utils.set_seed(global_config['seed'])
    device = C.get_device(global_config["device"])

    # data
    tp_df, fp_df, datadir, tp_fnames, tp_labels = C.get_metadata(config)
    sub_df, test_datadir = C.get_test_metadata(config)
    test_loader = C.get_loader(sub_df, test_datadir, config, phase="test")
    splitter = C.get_split(config)

    # mlflow logger
    config["mlflow"]["tags"]["timestamp"] = timestamp
    config["mlflow"]["tags"]["model_name"] = config["model"]["name"]
    config["mlflow"]["tags"]["loss_name"] = config["loss"]["name"]
    config["mlflow"]["tags"]["hash_value"] = hash_value
    mlf_logger = MLFlowLogger(
    experiment_name=config["mlflow"]["experiment_name"],
    tags=config["mlflow"]["tags"])

    # mlflowへconfigを渡す
    config_params = config.copy()
    del config_params['data'], config_params['globals'], config_params['mlflow']
    mlf_logger.log_hyperparams(config_params)

    all_preds = []  # 全体の結果を格納
    all_lwlrap_score = []  # val scoreを記録する用
    all_recall_score = []  # val scoreを記録する用
    all_precision_score = []  # val scoreを記録する用
    # for fold, (trn_idx, val_idx) in enumerate(splitter.split(df, y=df['species_id'])):
    for fold, (trn_idx, val_idx) in enumerate(splitter.split(tp_fnames, y=tp_labels)):
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
        train_fname = np.array(tp_fnames)[trn_idx]
        valid_fname = np.array(tp_fnames)[val_idx]
        trn_tp_df = tp_df[tp_df["recording_id"].isin(train_fname)] 
        val_df = tp_df[tp_df["recording_id"].isin(valid_fname)].reset_index(drop=True)
        trn_fp_df = fp_df[~fp_df["recording_id"].isin(valid_fname)]
        trn_df = pd.concat([trn_tp_df, trn_fp_df]).reset_index(drop=True)
        print("trainがvalに含まれているか: {}".format(set(trn_df["recording_id"].unique()).issubset(val_df["recording_id"].unique())))
        # trn_df = df.loc[trn_idx, :].reset_index(drop=True)
        # val_df = df.loc[val_idx, :].reset_index(drop=True)
        loaders = {
            phase: C.get_loader(df_, datadir, config, phase)
            for df_, phase in zip([trn_df, val_df], ["train", "valid"])
        }

        # callback
        checkpoint_callback = ModelCheckpoint(
            monitor=f'LWLRAP/val',
            mode='max',
            dirpath=output_dir,
            verbose=False,
            filename=f'{model_name}-{fold}')

        # model
        model = get_model(config)

        # load pretrained model
        if global_config['pretrained']:
            try:
                ckpt = torch.load(Path(global_config['pretrained_model_dir']) / f'{model_name}-{fold}-v0.ckpt') 
            except:
                ckpt = torch.load(Path(global_config['pretrained_model_dir']) / f'{model_name}-{fold}.ckpt')
            model.load_state_dict(ckpt['state_dict'])

        """
        ##############
        train part
        ##############
        """
        if global_config['only_pred']==False:
            # train
            trainer = pl.Trainer(
                logger=loggers, 
                checkpoint_callback=checkpoint_callback,
                max_epochs=global_config["max_epochs"],
                gpus=[0],
                fast_dev_run=global_config["debug"],
                deterministic=True)
            
            if not global_config['only_pred']:
                trainer.fit(model, train_dataloader=loaders['train'], val_dataloaders=loaders['valid'])

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
        make_oof(model, val_df, datadir, config, fold)
        make_test(model, test_loader, test_datadir, config, fold)
        make_fp(model, fp_df, datadir, config, fold)


if __name__ == '__main__':
    with utils.timer('Total time'):
        main()

