import argparse
import codecs
import json
import logging
import os
import random
import time
import requests
import subprocess
from datetime import datetime
import numpy as np
import torch
import yaml

from contextlib import contextmanager
from typing import Union, Optional
from pathlib import Path

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    msg = f"[{name}] start"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)


def get_logger(out_file=None):
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers = []
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if out_file is not None:
        fh = logging.FileHandler(out_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    logger.info("logger set up")
    return logger


def load_config(path: str):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


# get git hash value(short ver.)
def get_hash(config):
    if config['globals']["kaggle"]:
        # kaggle
        hash_value = None
    else:
        # local
        cmd = "git rev-parse --short HEAD"
        hash_value = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    
    return hash_value


def get_timestamp(config):
    # output config
    if config['globals']['timestamp']=='None':
        timestamp = datetime.today().strftime("%m%d_%H%M%S")
    else:
        timestamp = config['globals']['timestamp']
    
    if config["globals"]["debug"] == True:
        timestamp = "debug"
    return timestamp


# 任意のメッセージを通知する関数
def send_slack_message_notification(message):
    webhook_url = os.environ['SLACK_WEBHOOK_URL']  
    data = json.dumps({'text': message})
    headers = {'content-type': 'application/json'}
    requests.post(webhook_url, data=data, headers=headers)

# errorを通知する関数
def send_slack_error_notification(message):
    webhook_url = os.environ['SLACK_WEBHOOK_URL']  
    # no_entry_signは行き止まりの絵文字を出力
    data = json.dumps({"text":":no_entry_sign:" + message})  
    headers = {'content-type': 'application/json'}
    requests.post(webhook_url, data=data, headers=headers)