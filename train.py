import os
import torch
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from tqdm import tqdm
import hydra
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.models import get_model
import src.configuration as C
import src.utils as utils

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger


# 関数にconfig(cfg)を渡すデコレータ
# @hydra.main(config_path='./configs', config_name='ResNet001.yaml')
def main():
    warnings.filterwarnings('ignore')

    # config
    config_filename = 'ResNeSt003.yaml'
    config = utils.load_config(f"configs/{config_filename}")
    global_config = config['globals']

    # output config
    timestamp = datetime.today().strftime("%m%d_%H%M%S")

    if config["globals"]["debug"] == True:
        timestamp = "debug"
    output_dir = Path(global_config['output_dir']) / timestamp
    output_dir.mkdir(exist_ok=True, parents=True)

    # utils config
    logger = utils.get_logger(output_dir/ "output.log")
    utils.set_seed(global_config['seed'])
    device = C.get_device(global_config["device"])

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
        mlf_logger = MLFlowLogger(
        experiment_name=config["mlflow"]["experiment_name"],
        tags=config["mlflow"]["tags"])
        loggers.append(mlf_logger)


    model_name = config["model"]['name']
    tb_logger = TensorBoardLogger(save_dir=output_dir, name=model_name)
    loggers.append(tb_logger)
    
    # data
    df, datadir = C.get_metadata(config)
    sub_df, test_datadir = C.get_test_metadata(config)
    splitter = C.get_split(config)

    """
    ##############
    train part
    ##############
    """

    for fold, (trn_idx, val_idx) in enumerate(splitter.split(df, y=df['species_id'])):
        # 指定したfoldのみループを回す
        if fold not in global_config['folds']:
            continue

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
        early_stop_callback = EarlyStopping(monitor='val_loss')
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=output_dir,
            verbose=True,
            filename=f'{model_name}-{fold}')
       
        # model
        model = get_model(config)
        # train
        trainer = pl.Trainer(
            logger=loggers, 
            checkpoint_callback=checkpoint_callback,
            # callbacks=[early_stop_callback],
            max_epochs=global_config["num_epochs"],
            gpus=int(torch.cuda.is_available()),
            fast_dev_run=global_config["debug"])
        
        trainer.fit(model, train_dataloader=loaders['train'], val_dataloaders=loaders['valid'])

    
    """
    ##############
    inference part
    ##############
    """

    total_preds = []  # 全体の結果を格納
    for fold in global_config["folds"]:

        # load checkpoint
        model = get_model(config)
        ckpt = torch.load(checkpoint_callback.best_model_path)  # TODO foldごとのモデルを取得できるようにする
        model.load_state_dict(ckpt['state_dict'])

        # 推論結果出力
        model.eval().to(device)
        test_loader = C.get_loader(sub_df, test_datadir, config, phase="test")
        preds = []
        with torch.no_grad():
    
            # xが複数の場合
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
            
            """
            # xが１つの場合
            for x, y in tqdm(test_loader):
                output = model(x)
                output = output[config["globals"]["output_type"]]
                preds.append(output)

            preds = np.vstack(preds) 
            """

        total_preds.append(preds)  # foldの予測結果を格納

    sub_preds = np.mean(total_preds, axis=0)  # foldで平均を取る
    sub_df.iloc[:, 1:] = sub_preds
    sub_df.to_csv(output_dir / "submission.csv", index=False)
        

if __name__ == '__main__':
    main()

