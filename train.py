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
from pytorch_lightning.metrics import F1, Accuracy, Recall

os.environ['NUMEXPR_MAX_THREADS'] = '24'


def valid_step(model, val_df, loaders, config, output_dir, fold):
    # ckptのモデルでoof出力
    preds = []
    scores = []
    output_key = config['model']['output_key']
    device = 'cuda'
    with torch.no_grad():
        # xは複数のlist
        for x_list, y in tqdm(loaders['valid']):
            batch_size = x_list.shape[0]
            x = x_list.view(-1, x_list.shape[2], x_list.shape[3], x_list.shape[4])  # batch>1でも可
            x = x.to(device)
            
            if "SED" in config["model"]["name"]:
                output = model.model(x)
                output = output[output_key]
            output = output.view(batch_size, -1, 24)  # 24=num_classes
            pred = torch.max(output, dim=1)[0]  # 1次元目(分割sしたやつ)で各クラスの最大を取得
            y = y.to(device)
            recall = Recall()
            score = recall(pred.sigmoid(), y) 
            scores.append(score)
            pred = torch.argsort(pred, dim=-1, descending=True)
            preds.append(pred.detach().cpu().numpy())

        # log
        valid_score = np.mean(scores, axis=0)  
    
        return valid_score


def test_step(model, sub_df, test_loader, config, output_dir, fold):
    # 推論結果出力
    
    preds = []
    with torch.no_grad():
        # xは複数のlist
        for x_list, _ in tqdm(test_loader):
            batch_size = x_list.shape[0]
            x = x_list.view(-1, x_list.shape[2], x_list.shape[3], x_list.shape[4])  # batch>1でも可
            x = x.to(config["globals"]["device"])
            if "SED" in config["model"]["name"]:
                output = model.model(x)
                output = output["logit"]
            output = output.view(batch_size, -1, 24)  # 24=num_classes
            pred = torch.max(output, dim=1)[0]  # 1次元目(分割sしたやつ)で各クラスの最大を取得
            pred = pred.detach().cpu().numpy()
            preds.append(pred)
        
        preds = np.vstack(preds)  # 全データを１つのarrayにつなげてfoldの予測とする
        fold_df = sub_df.copy()
        fold_df.iloc[:, 1:] = preds
        fold_df.to_csv(output_dir / f'fold{fold}.csv', index=False)
    return preds 


def main():
    warnings.filterwarnings('ignore')

    # config
    config_filename = 'EfficientNetSED001.yaml'
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
    df, datadir = C.get_metadata(config)
    sub_df, test_datadir = C.get_test_metadata(config)
    test_loader = C.get_loader(sub_df, test_datadir, config, phase="test")
    splitter = C.get_split(config)


    all_preds = []  # 全体の結果を格納
    all_lwlrap_score = []  # val scoreを記録する用
    for fold, (trn_idx, val_idx) in enumerate(splitter.split(df, y=df['species_id'])):
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
        trn_df = df.loc[trn_idx, :].reset_index(drop=True)
        val_df = df.loc[val_idx, :].reset_index(drop=True)
        loaders = {
            phase: C.get_loader(df_, datadir, config, phase)
            for df_, phase in zip([trn_df, val_df], ["train", "valid"])
        }

        # callback
        checkpoint_callback = ModelCheckpoint(
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
        
        # valid
        valid_score = valid_step(model, val_df, loaders, config, output_dir, fold)
        all_lwlrap_score.append(valid_score)

        # test
        # preds = test_step(model, sub_df, test_loader, config, output_dir, fold)
        # all_preds.append(preds)  # foldの予測結果を格納
        utils.send_slack_message_notification(f'[FINISH] fold{fold}-lwlrap:{valid_score:.3f}')
     

if __name__ == '__main__':
    with utils.timer('Total time'):
        main()

