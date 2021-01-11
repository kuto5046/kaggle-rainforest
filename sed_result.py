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
import yaml
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

os.environ['NUMEXPR_MAX_THREADS'] = '24'

config_str = """
globals:
  seed: 42
  device: cuda
  max_epochs: 30
  output_dir: output
  timestamp: None
  only_pred: False
  folds:
    - 0
    - 1
    - 2
    - 3
    - 4
  debug: True
  kaggle: False

mlflow:
    experiment_name: rainforest
    tags:
      timestamp: None
      model_name: None
      loss_name: None
      hash_value: None
      message: make best model

data:
  root: ./
  use_train_data:
    - tp
   #  - fp 
  train_tp_df_path: input/rfcx-species-audio-detection/train_tp.csv
  train_fp_df_path: input/rfcx-species-audio-detection/train_fp.csv
  train_audio_path: input/rfcx-species-audio-detection/train
  test_audio_path: input/rfcx-species-audio-detection/test
  sub_df_path: input/rfcx-species-audio-detection/sample_submission.csv


dataset:
  name: SpectrogramDataset
  height: 224
  width: 400
  period: 10
  shift_time: 10
  strong_label_prob: 1.0
  params:
    melspec:
      n_fft: 2048
      n_mels: 128
      fmin: 80
      fmax: 15000
      power: 2.0
    pcen:
      gain: 0.98
      bias: 2
      power: 0.5
      time_constant: 0.4
      eps: 0.000001

loss:
  name: LSEPStableLoss  # LSEPStableLoss LSEPLoss  CustomBCELoss
  params:
    output_key: logit

optimizer:
  name: Adam
  params:
    lr: 0.001

scheduler:
  name: CosineAnnealingLR
  params:
    T_max: 10

split:
  name: StratifiedKFold
  params:
    n_splits: 5
    random_state: 42
    shuffle: True

model:
  name: EfficientNetSED
  output_key: logit
  params:
    base_model_name: efficientnet-b2
    pretrained: True
    num_classes: 24

loader:
  train:
    batch_size: 20
    shuffle: True
    num_workers: 10
  valid:
    batch_size: 20
    shuffle: False
    num_workers: 10
  test:
    batch_size: 20
    shuffle: False
    num_workers: 10

transforms:
  train:

  valid:

  test:

mixup:
  flag: False
  alpha: 0.2
  prob: 0.5
"""

def main():
    warnings.filterwarnings('ignore')

    # config
    config = yaml.safe_load(config_str)
    global_config = config['globals']
    timestamp = "0110_121231"
    config['globals']['timestamp'] = timestamp

    input_dir = Path(global_config['output_dir']) / timestamp

    # utils config
    utils.set_seed(global_config['seed'])
    device = C.get_device('cpu')

    # data
    _, datadir = C.get_metadata(config)
    output_dir = Path(f'./output/sed_result/{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    for fold in range(5):
        # model
        model = get_model(config)
        model_name = config['model']['name']
        # load model
        try:
            ckpt = torch.load(input_dir / f'{model_name}-{fold}-v0.ckpt')  # TODO foldごとのモデルを取得できるようにする
        except:
            ckpt = torch.load(input_dir / f'{model_name}-{fold}.ckpt')  # TODO foldごとのモデルを取得できるようにする
        model.load_state_dict(ckpt['state_dict'])
        model.eval().to(device)

        oof = pd.read_csv(input_dir / f'oof_fold{fold}.csv')
        dataloader = C.get_loader(oof, datadir, config, phase="valid")
        with torch.no_grad():
            # xは複数のlist
            for idx, (x_list, y) in enumerate(dataloader):
                batch_size = x_list.shape[0]
                if batch_size == config['loader']['valid']['batch_size']:
                    recording_ids = oof.loc[batch_size*idx:batch_size*(idx+1), 'recording_id'].values
                else:
                    recording_ids = oof.loc[-batch_size:, 'recording_id'].values
                x = x_list.view(-1, x_list.shape[2], x_list.shape[3], x_list.shape[4])  # batch>1でも可
                
                outputs = model.model(x)
                framewise_outputs = outputs['framewise_output']
                framewise_outputs = framewise_outputs.view(batch_size, -1, 24)  # 24=num_classes
                framewise_outputs = framewise_outputs.numpy()
                y = y.numpy()
                for i, output in enumerate(framewise_outputs):
                    recording_id = recording_ids[i]
                    species_id = np.where(y[i]==1)[0][0]
                    np.save(output_dir / f'{recording_id}-{species_id}', output)



        

if __name__ == '__main__':
    with utils.timer('Total time'):
        main()


