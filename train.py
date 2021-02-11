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


def valid_step(model, val_df, loaders, config, output_dir, fold):
    # ckptのモデルでoof出力
    preds = []
    lwlrap_scores = []
    recall_scores = []
    precision_scores = []
    recall = Recall(num_classes=24, multilabel=True)
    precision = Precision(num_classes=24, multilabel=True)
    output_key = config['model']['output_key']
    with torch.no_grad():
        # xは複数のlist
        for x_list, y in tqdm(loaders['valid']):
            batch_size = x_list.shape[0]
            x = x_list.view(-1, x_list.shape[2], x_list.shape[3], x_list.shape[4])
            x = x.to("cuda")

            output = model.model(x, None, do_mixup=False)
            pred = output[output_key]
            
            # 同じファイルで集約
            pred = pred.view(batch_size, -1, 24)  # (batch, 8, 24)

            posi_mask = (y > 0).float()
            y = (y * posi_mask).sum(axis=1)  # y3 (batch, 24)
            y = (y > 0).float()
            pred = pred.max(axis=1)[0]  # (batch, 24)
            pred = pred.to('cpu')
            lwlrap_score = LWLRAP(pred, y)
            recall_score = recall(pred.sigmoid(), y)
            precision_score = precision(pred.sigmoid(), y)
            lwlrap_scores.append(lwlrap_score)
            recall_scores.append(recall_score)
            precision_scores.append(precision_score)
            pred = torch.argsort(pred, dim=-1, descending=True)
            preds.append(pred.detach().cpu().numpy())

        # log
        lwlrap_score = np.mean(lwlrap_scores, axis=0)
        recall_score = np.mean(recall_scores, axis=0)
        precision_score = np.mean(precision_scores, axis=0)
    
        return lwlrap_score, recall_score, precision_score


def test_step(model, sub_df, test_loader, config, output_dir, fold):
    # 推論結果出力
    
    preds = []
    with torch.no_grad():
        # xは複数のlist
        for x_list, _ in tqdm(test_loader):
            batch_size = x_list.shape[0]
            x = x_list.view(-1, x_list.shape[2], x_list.shape[3], x_list.shape[4])  # batch>1でも可
            x = x.to(config["globals"]["device"])

            output = model.model(x, None, do_mixup=False)
            output = output["logit"]
            output = output.view(batch_size, -1, 24)  # 24=num_classes
            pred = torch.max(output, dim=1)[0]  # 1次元目(分割sしたやつ)で各クラスの最大を取得
            pred = pred.sigmoid().detach().cpu().numpy()
            preds.append(pred)
        
        preds = np.vstack(preds)  # 全データを１つのarrayにつなげてfoldの予測とする
        fold_df = sub_df.copy()
        fold_df.iloc[:, 1:] = preds
        fold_df.to_csv(output_dir / f'fold{fold}.csv', index=False)
    return preds


def main():
    warnings.filterwarnings('ignore')

    # config
    config_filename = 'EfficientNetSED003.yaml'
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
    df, datadir, tp_fnames, tp_labels, fp_fnames, fp_labels = C.get_metadata(config)
    sub_df, test_datadir = C.get_test_metadata(config)
    test_loader = C.get_loader(sub_df, test_datadir, config, phase="test")
    splitter1 = C.get_split(config)
    splitter2 = C.get_split(config)

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

    # Make CV
    tp_cv = [(np.array(tp_fnames)[train_index], np.array(tp_fnames)[valid_index]) for train_index, valid_index in splitter1.split(tp_fnames, tp_labels)]
    fp_cv = [(np.array(fp_fnames)[train_index], np.array(fp_fnames)[valid_index]) for train_index, valid_index in splitter2.split(fp_fnames, fp_labels)]

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
                deterministic=True,
                precision=16)
            
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
        lwlrap_score, recall_score, precision_score = valid_step(model, val_df, loaders, config, output_dir, fold)
        mlf_logger.log_metrics({f'LWLRAP/fold{fold}':lwlrap_score}, step=None)
        all_lwlrap_score.append(lwlrap_score)
        all_recall_score.append(recall_score)
        all_precision_score.append(precision_score)

        # test
        preds = test_step(model, sub_df, test_loader, config, output_dir, fold)
        all_preds.append(preds)  # foldの予測結果を格納
        utils.send_slack_message_notification(f'[FINISH] fold{fold}-lwlrap:{lwlrap_score:.3f}')

        # oof
        fold_oof = make_oof(model, val_df, datadir, config, fold, output_dir)
        all_oof = pd.concat([all_oof, fold_oof])
        make_test(model, test_loader, test_datadir, config, fold, output_dir)


    # ループ抜ける
    # save oof 
    all_oof = all_oof.reset_index(drop=True)
    all_oof.to_csv(output_dir / "all_oof.csv", index=False)


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
        
        val_recall_score = 0
        val_precision_score = 0
        for rec, prec in zip(all_recall_score, all_precision_score):
            val_recall_score += rec
            val_precision_score += prec
        
        val_recall_score = val_recall_score / len(all_recall_score)
        val_precision_score = val_precision_score / len(all_precision_score)
    else:
        raise NotImplementedError

    mlf_logger.log_metrics({f'LWLRAP/all':val_lwlrap_score}, step=None)
    mlf_logger.log_metrics({f'Recall/all':val_recall_score}, step=None)
    mlf_logger.log_metrics({f'Precision/all':val_precision_score}, step=None)
    mlf_logger.log_metrics({f'LWLRAP/LB_Score': 0.0}, step=None)
    mlf_logger.finalize()

    sub_df.iloc[:, 1:] = sub_preds
    sub_df.to_csv(output_dir / "submission.csv", index=False)
        

if __name__ == '__main__':
    with utils.timer('Total time'):
        main()

