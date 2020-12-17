import os
import torch
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

import tqdm
import hydra
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.models import ResNet
import src.configuration as C
import src.utils as utils

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger


class Learner(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.model = model
        

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        output = output[self.config["globals"]["output_type"]]
        criterion = C.get_criterion(self.config)
        loss = criterion(output, y)
        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        output = output[self.config["globals"]["output_type"]]
        criterion = C.get_criterion(self.config)
        loss = criterion(output, y)
        self.log('val_loss', loss)

        return loss

    # def test_step(self, batch, batch_idx):
    #     x, _ = batch
    #     output = self.forward(x)
    #     output = output[self.config["globals"]["output_type"]]

    #     return output


    def configure_optimizers(self):
        optimizer = C.get_optimizer(self.model, self.config)
        scheduler = C.get_scheduler(optimizer, self.config)
        return [optimizer], [scheduler]

# 関数にconfig(cfg)を渡すデコレータ
# @hydra.main(config_path='./configs', config_name='ResNet001.yaml')
def main():
    warnings.filterwarnings('ignore')

    # config
    config = utils.load_config("configs/ResNet001.yaml")
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

    # mlflow
    os.makedirs(config["mlflow"]["tracking_uri"], exist_ok=True)
    mlf_logger = MLFlowLogger(
    experiment_name=config["mlflow"]["experiment_name"],
    tracking_uri=config["mlflow"]["tracking_uri"])
    
    # data
    df, datadir = C.get_metadata(config)
    sub_df, test_datadir = C.get_test_metadata(config)
    splitter = C.get_split(config)
    

    for fold, (trn_idx, val_idx) in enumerate(splitter.split(df, y=df['species_id'])):
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
        model_name = global_config['model_name']
        # tb_logger = TensorBoardLogger(save_dir=output_dir, name=model_name, version=f'fold_{fold + 1}')
        early_stop_callback = EarlyStopping(monitor='val_loss')

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=output_dir,
            filename=f'{model_name}-{fold:1d}')

       
        # model
        model = ResNet(config)
        learner = Learner(model, config)

        # train
        trainer = pl.Trainer(
            logger=[mlf_logger], 
            checkpoint_callback=checkpoint_callback,
            callbacks=[early_stop_callback],
            max_epochs=global_config["num_epochs"],
            gpus=int(torch.cuda.is_available()),
            fast_dev_run=global_config["debug"])
        
        trainer.fit(learner, train_dataloader=loaders['train'], val_dataloaders=loaders['valid'])


    # load checkpoint
    model = Learner(None, config)
    checkpoint = torch.load('/content/lightning_logs/version_0/checkpoints/_ckpt_epoch_6.ckpt')
    model.load_state_dict(checkpoint['state_dict'])

    # 推論結果出力
    preds = []
    model.eval()
    test_loader = C.get_testloader(sub_df, test_datadir, config, phase="test")
    with torch.no_grad():
        for x in tqdm(test_loader):
            x = x.to(device)
            output = model(x)
            output = output[config["globals"]["output_type"]] 
            preds.append(output.detach().cpu().numpy())
    preds = np.vstack(preds)

    # make submission file
    sub_df
    sub.head()


if __name__ == '__main__':
    main()

