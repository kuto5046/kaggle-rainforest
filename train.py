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

os.environ['NUMEXPR_MAX_THREADS'] = '24'


def valid_step(model, val_df, loaders, config, output_dir, fold):
    # ckptのモデルでoof出力
    preds = []
    scores = []
    output_key = config['model']['output_key']
    with torch.no_grad():
        # xは複数のlist
        for x_list, y in tqdm(loaders['valid']):
            batch_size = x_list.shape[0]
            x = x_list.view(-1, x_list.shape[2], x_list.shape[3], x_list.shape[4])  # batch>1でも可
            x = x.to(config["globals"]["device"])
            
            if "SED" in config["model"]["name"]:
                output = model.model(x)
                output = output[output_key]
            output = output.view(batch_size, -1, 24)  # 24=num_classes
            pred = torch.max(output, dim=1)[0]  # 1次元目(分割sしたやつ)で各クラスの最大を取得
            score = LWLRAP(pred, y)
            scores.append(score)
            pred = torch.argsort(pred, dim=-1, descending=True)
            preds.append(pred.detach().cpu().numpy())

        # log
        lwlrap_score = np.mean(scores, axis=0)

        # make oof
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
    
        return lwlrap_score


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
    tp_df, re_df, datadir = C.get_metadata(config)
    df = pd.concat([tp_df, re_df])
    df = df[df['data_type'].isin(config['data']['use_train_data'])].reset_index(drop=True)

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
        # reのデータはtpのデータで上書きしたい
        if 'tp' in config['data']['use_train_data']:
            val_df = val_df[val_df['data_type']=='tp']  # valid scoreの質を担保するために追加(tpデータと一緒に学習する場合のみOK)
        else:
            # valデータに分類されたものでrecording_idが同じものをtp_dfから抽出
            val_df = tp_df[tp_df['recording_id'].isin(val_df['recording_id'].unique())]
        
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
            ckpt = torch.load(output_dir / f'{model_name}-{fold}-v0.ckpt')  # TODO foldごとのモデルを取得できるようにする
        except:
            ckpt = torch.load(output_dir / f'{model_name}-{fold}.ckpt')  # TODO foldごとのモデルを取得できるようにする
        model.load_state_dict(ckpt['state_dict'])
        model.eval().to(device)
        
        # valid
        lwlrap_score = valid_step(model, val_df, loaders, config, output_dir, fold)
        mlf_logger.log_metrics({f'LWLRAP/fold{fold}':lwlrap_score}, step=None)
        all_lwlrap_score.append(lwlrap_score)

        # test
        preds = test_step(model, sub_df, test_loader, config, output_dir, fold)
        all_preds.append(preds)  # foldの予測結果を格納
        utils.send_slack_message_notification(f'[FINISH] fold{fold}-lwlrap:{lwlrap_score:.3f}')


    # ループ抜ける
    # final logger 
    # valの結果で加重平均
    mean_method = 'weight_mean'
    if mean_method == 'mean':
        val_lwlrap_score = np.mean(all_lwlrap_score, axis=0)
        sub_preds = np.mean(all_preds, axis=0)  # foldで平均を取る
    elif mean_method == 'weight_mean':
        weights = []
        val_lwlrap_score = 0
        for i, score in enumerate(all_lwlrap_score):
            weight = score / np.sum(all_lwlrap_score)
            val_lwlrap_score += all_lwlrap_score[i] * weight
            weights.append(weight)
        # for submission
        sub_preds = 0
        for i, pred in enumerate(all_preds):
            sub_preds += pred * weights[i]
    else:
        raise NotImplementedError

    mlf_logger.log_metrics({f'LWLRAP/all':val_lwlrap_score}, step=None)
    mlf_logger.log_metrics({f'LWLRAP/LB_Score': 0.0}, step=None)
    mlf_logger.finalize()

    sub_df.iloc[:, 1:] = sub_preds
    sub_df.to_csv(output_dir / "submission.csv", index=False)
        

if __name__ == '__main__':
    with utils.timer('Total time'):
        main()

