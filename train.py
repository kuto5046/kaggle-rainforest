import os
import torch
import warnings
from pathlib import Path
from datetime import datetime

import hydra
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
        output = self.foward(x)
        criterion = C.get_criterion(self.config)
        loss = criterion(output, y)
        self.logger.experiment.whatever_ml_flow_supports()

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        criterion = C.get_criterion(self.config)
        loss = criterion(output, y)
        self.logger.experiment.whatever_ml_flow_supports()
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = C.get_optimizer(self.model, self.config)
        scheduler = C.get_scheduler(optimizer, self.config)
        return optimizer, scheduler

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
    mlf_logger = MLFlowLogger(
    experiment_name=config["mlflow"]["experiment_name"],
    tracking_uri=config["mlflow"]["tracking_uri"],
    tags=config["mlflow"]["tags"])

    # data
    df, datadir = C.get_metadata(config)
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
        tb_logger = TensorBoardLogger(save_dir=output_dir, name=global_config['model_name'], version=f'fold_{fold + 1}')
        early_stop_callback = EarlyStopping()
        checkpoint_callback = ModelCheckpoint(
            filepath=tb_logger.log_dir + "/{epoch:02d}-{val_metric:.4f}", 
            monitor='val_metric')
        
        # model
        model = ResNet(config)
        learner = Learner(model, config)

        # train
        trainer = pl.Trainer(
            logger=[tb_logger, mlf_logger], 
            checkpoint_callback=checkpoint_callback,
            callbacks=[early_stop_callback],
            gpus=int(torch.cuda.is_available()))
        
        trainer.fit(learner, train_dataloader=loaders['train'], val_dataloaders=loaders['valid'])


if __name__ == '__main__':
    main()

