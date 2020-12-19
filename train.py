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

from src.models import get_model
import src.configuration as C
import src.utils as utils

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger


# LRAP. Instance-level average
# Assume float preds [BxC], labels [BxC] of 0 or 1
def LRAP(preds, labels):
    # Ranks of the predictions
    ranked_classes = torch.argsort(preds, dim=-1, descending=True)
    # i, j corresponds to rank of prediction in row i
    class_ranks = torch.zeros_like(ranked_classes)
    for i in range(ranked_classes.size(0)):
        for j in range(ranked_classes.size(1)):
            class_ranks[i, ranked_classes[i][j]] = j + 1
    # Mask out to only use the ranks of relevant GT labels
    ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
    # All the GT ranks are in front now
    sorted_ground_truth_ranks, _ = torch.sort(ground_truth_ranks, dim=-1, descending=False)
    pos_matrix = torch.tensor(np.array([i+1 for i in range(labels.size(-1))])).unsqueeze(0)
    score_matrix = pos_matrix / sorted_ground_truth_ranks
    score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
    scores = score_matrix * score_mask_matrix
    score = (scores.sum(-1) / labels.sum(-1)).mean()
    return score.item()

# label-level average
# Assume float preds [BxC], labels [BxC] of 0 or 1
def LWLRAP(preds, labels):
    # Ranks of the predictions
    ranked_classes = torch.argsort(preds, dim=-1, descending=True)
    # i, j corresponds to rank of prediction in row i
    class_ranks = torch.zeros_like(ranked_classes)
    for i in range(ranked_classes.size(0)):
        for j in range(ranked_classes.size(1)):
            class_ranks[i, ranked_classes[i][j]] = j + 1
    # Mask out to only use the ranks of relevant GT labels
    ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
    # All the GT ranks are in front now
    sorted_ground_truth_ranks, _ = torch.sort(ground_truth_ranks, dim=-1, descending=False)
    # Number of GT labels per instance
    num_labels = labels.sum(-1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pos_matrix = torch.tensor(np.array([i+1 for i in range(labels.size(-1))])).unsqueeze(0).to(device)
    score_matrix = pos_matrix / sorted_ground_truth_ranks
    score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
    scores = score_matrix * score_mask_matrix
    score = scores.sum() / labels.sum()
    return score.item()

# Sample usage
# y_true = torch.tensor(np.array([[1, 1, 0], [1, 0, 1], [0, 0, 1]]))
# y_score = torch.tensor(np.random.randn(3, 3))
# print(LRAP(y_score, y_true), LWLRAP(y_score, y_true))


class Learner(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = get_model(config)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        output = output[self.config["globals"]["output_type"]]
        criterion = C.get_criterion(self.config)
        loss = criterion(output, y)
        lwlrap = LWLRAP(output, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_LWLRAP', lwlrap, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        output = output[self.config["globals"]["output_type"]]
        criterion = C.get_criterion(self.config)
        loss = criterion(output, y)
        lwlrap = LWLRAP(output, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_LWLRAP', lwlrap, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    

    def configure_optimizers(self):
        optimizer = C.get_optimizer(self.model, self.config)
        scheduler = C.get_scheduler(optimizer, self.config)
        return [optimizer], [scheduler]


# 関数にconfig(cfg)を渡すデコレータ
# @hydra.main(config_path='./configs', config_name='ResNet001.yaml')
def main():
    warnings.filterwarnings('ignore')

    # config
    config = utils.load_config("configs/ResNeSt001.yaml")
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
    # os.makedirs(config["mlflow"]["tracking_uri"], exist_ok=True)
    config["mlflow"]["tags"]["timestamp"] = timestamp
    mlf_logger = MLFlowLogger(
    experiment_name=config["mlflow"]["experiment_name"],
    tags=config["mlflow"]["tags"])

    model_name = global_config['model_name']
    tb_logger = TensorBoardLogger(save_dir=output_dir, name=model_name)
    
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
        early_stop_callback = EarlyStopping(monitor='val_loss')
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=output_dir,
            verbose=True,
            filename=f'{model_name}-{fold}')

       
        # model
        learner = Learner(config)

        # train
        trainer = pl.Trainer(
            logger=[tb_logger, mlf_logger], 
            checkpoint_callback=checkpoint_callback,
            callbacks=[early_stop_callback],
            max_epochs=global_config["num_epochs"],
            gpus=int(torch.cuda.is_available()),
            fast_dev_run=global_config["debug"])
        
        trainer.fit(learner, train_dataloader=loaders['train'], val_dataloaders=loaders['valid'])

    
    """
    ##############
    inference part
    ##############
    """
    preds = []

    for i in global_config["folds"]:
        
        # load checkpoint
        model = Learner(config)
        ckpt = torch.load(checkpoint_callback.best_model_path)
        model.load_state_dict(ckpt['state_dict'])

        # 推論結果出力
        pred = []
        model.eval().to(device)
        test_loader = C.get_testloader(sub_df, test_datadir, config, phase="test")
        
        with torch.no_grad():
            for x in tqdm(test_loader):
                x = x.to(device)
                output = model(x)
                output = output[config["globals"]["output_type"]] 
                pred.append(output.detach().cpu().numpy())
        pred = np.vstack(pred)
        preds.append(pred)
    
    sub_preds = np.mean(preds, axis=0)
    sub_df.iloc[:, 1:] = sub_preds
    sub_df.to_csv(output_dir / "submission.csv", index=False)


if __name__ == '__main__':
    main()

