import os
import torch
import subprocess
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

from src.models import get_model
import src.configuration as C
import src.utils as utils
from src.metric import LWLRAP
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

os.environ['NUMEXPR_MAX_THREADS'] = '24'

# 関数にconfig(cfg)を渡すデコレータ
# @hydra.main(config_path='./configs', config_name='ResNet001.yaml')
def main():
    warnings.filterwarnings('ignore')

    # config
    config_filename = 'ResNeSt003.yaml'
    config = utils.load_config(f"configs/{config_filename}")
    global_config = config['globals']

    # get git hash value(short ver.)
    if global_config["kaggle"]:
        # kaggle
        hash_value = None
    else:
        # local
        cmd = "git rev-parse --short HEAD"
        hash_value = subprocess.check_output(cmd.split()).strip().decode('utf-8')

    # output config
    if config['globals']['timestamp']=='None':
        timestamp = datetime.today().strftime("%m%d_%H%M%S")
    else:
        timestamp = config['globals']['timestamp']
    
    if config["globals"]["debug"] == True:
        timestamp = "debug"
    output_dir = Path(global_config['output_dir']) / timestamp
    output_dir.mkdir(exist_ok=True, parents=True)

    # utils config
    logger = utils.get_logger(output_dir/ "output.log")
    logger.info(config)
    utils.set_seed(global_config['seed'])
    device = C.get_device(global_config["device"])

    # data
    df, datadir = C.get_metadata(config)
    sub_df, test_datadir = C.get_test_metadata(config)
    splitter = C.get_split(config)

    total_preds = []  # 全体の結果を格納

    for fold, (trn_idx, val_idx) in enumerate(splitter.split(df, y=df['species_id'])):
        # 指定したfoldのみループを回す
        if fold not in global_config['folds']:
            continue

        # logger
        loggers = []
        if global_config["debug"]==True:
            mlf_logger = None
        else:
            # os.makedirs(config["mlflow"]["tracking_uri"], exist_ok=True)
            config["mlflow"]["tags"]["timestamp"] = timestamp
            config["mlflow"]["tags"]["config_filename"] = config_filename
            config["mlflow"]["tags"]["model_name"] = config["model"]["name"]
            config["mlflow"]["tags"]["loss_name"] = config["loss"]["name"]
            config["mlflow"]["tags"]["hash_value"] = hash_value
            config["mlflow"]["tags"]["fold"] = fold
            mlf_logger = MLFlowLogger(
            experiment_name=config["mlflow"]["experiment_name"],
            tags=config["mlflow"]["tags"])
            loggers.append(mlf_logger)

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
        # early_stop_callback = EarlyStopping(monitor='val_loss')
        checkpoint_callback = ModelCheckpoint(
            monitor=f'loss/val',
            mode='min',
            dirpath=output_dir,
            verbose=False,
            filename=f'{model_name}-{fold}')
        
        """
        ##############
        train part
        ##############
        """
        # model
        model = get_model(config)

        # train
        trainer = pl.Trainer(
            logger=loggers, 
            checkpoint_callback=checkpoint_callback,
            max_epochs=global_config["max_epochs"],
            gpus=-1,
            fast_dev_run=global_config["debug"])
        
        if not global_config['only_pred']:
            trainer.fit(model, train_dataloader=loaders['train'], val_dataloaders=loaders['valid'])

        """
        ##############
        inference part
        ##############
        """
        # load checkpoint
        # model = get_model(config, fold)
        try:
            ckpt = torch.load(output_dir / f'{model_name}-{fold}-v0.ckpt')  # TODO foldごとのモデルを取得できるようにする
        except:
            ckpt = torch.load(output_dir / f'{model_name}-{fold}.ckpt')  # TODO foldごとのモデルを取得できるようにする
        model.load_state_dict(ckpt['state_dict'])
        model.eval().to(device)
    
        # ckptのモデルでoof出力
        preds = []
        with torch.no_grad():
            # xは複数のlist
            for x_list, y in tqdm(loaders['valid']):
                batch_size = x_list.shape[0]
                x = x_list.view(-1, x_list.shape[2], x_list.shape[3], x_list.shape[4])  # batch>1でも可
                x = x.to(config["globals"]["device"])
                output = model(x)
                output = output.view(batch_size, -1, 24)  # 24=num_classes
                pred = torch.max(output, dim=1)[0]  # 1次元目(分割sしたやつ)で各クラスの最大を取得
                pred = torch.argsort(pred, dim=-1, descending=True)
                pred = pred.detach().cpu().numpy()
                preds.append(pred)
            
            # TODO metric指標を出したいがその場合pytorchで扱う必要あり
            oof_df = val_df.copy()
            pred_columns = [f'rank_{i}' for i in range(24)]
            for col in pred_columns:
                oof_df[col] = 0
            oof_df.loc[:, 'rank_0':] = np.vstack(preds) # 全データを１つのarrayにつなげてfoldの予測とする
            
            # pecies_idに対応する予測結果のrankingを取り出す
            rankings = []
            top_ids = []
            for _, raw in oof_df.iterrows():
                species_id = raw['species_id']
                rankings.append(int(list(raw['rank_0':][raw==species_id].index)[0][5:]))  # speicesの順番(rank番号を取得)
                top_ids.append(np.argmin(raw['rank_0':].values))
            oof_df['ranking'] = rankings
            oof_df['top_id'] = top_ids
            oof_df.to_csv(output_dir / f'oof_fold{fold}.csv', index=False)

        if global_config["debug"]==False:
            # TODO mlflowのmetricをckptのモデルのscoreで出力する
            pass

        # 推論結果出力
        test_loader = C.get_loader(sub_df, test_datadir, config, phase="test")
        preds = []
        with torch.no_grad():
            # xは複数のlist
            for x_list, _ in tqdm(test_loader):
                batch_size = x_list.shape[0]
                x = x_list.view(-1, x_list.shape[2], x_list.shape[3], x_list.shape[4])  # batch>1でも可
                x = x.to(config["globals"]["device"])
                output = model(x)
                output = output.view(batch_size, -1, 24)  # 24=num_classes
                pred = torch.max(output, dim=1)[0]  # 1次元目(分割sしたやつ)で各クラスの最大を取得
                pred = pred.detach().cpu().numpy()
                preds.append(pred)
            
            preds = np.vstack(preds)  # 全データを１つのarrayにつなげてfoldの予測とする
            fold_df = sub_df.copy()
            fold_df.iloc[:, 1:] = preds
            fold_df.to_csv(output_dir / f'fold{fold}.csv', index=False)
        total_preds.append(preds)  # foldの予測結果を格納

    sub_preds = np.mean(total_preds, axis=0)  # foldで平均を取る
    sub_df.iloc[:, 1:] = sub_preds
    sub_df.to_csv(output_dir / "submission.csv", index=False)
        

if __name__ == '__main__':
    with utils.timer('Total time'):
        main()

