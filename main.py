#!/usr/bin/env python3

from typing import Dict

import torch

import json
import os
import multiprocessing
import random
import argparse
import numpy as np

from common import *
from data import *
from train import *

seed = 777
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)


def train_task(data_tag, data_path, log_path, fig_path, gpu_id, waterline, 
                train_test_params, data_set_params):
    
    dataset = load_data(data_tag, data_path, shuffle_seed=seed, fig_path=fig_path, **data_set_params)
    train_test(data_tag, log_path, fig_path, *dataset, gpu_id, waterline, **train_test_params)

def validate_json_config(jin: Dict) -> bool:
    key_list = ['data_path', 'log_path', 'fig_path', 'data_tag', 
                    'gpu_enable', 'data_construct_param', 'train_param']
    for k in key_list:
        if k not in jin:
            print(f'Key {k} is missed in configuration.')
            return False
    
    try:
        if not os.path.exists(jin['log_path']):
            os.makedirs(jin['log_path'])
        if not os.path.exists(jin['fig_path']):
            os.makedirs(jin['fig_path'])
        if 'model_save' in jin and not os.path.exists(jin['model_save']):
            os.makedirs(jin['model_save'])
    except Exception as e:
        logging.error(f'Exceprtion in create folder' + e)
        return False
    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ExoSphere: detect DDoS at L2.')
    parser.add_argument('-c', '--config', type=str, default='./config.json', help='Configuration file.')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        jin = json.load(f)
    if not validate_json_config(jin):
        exit(-1)
    
    pve = []
    for i, tag in enumerate(jin['data_tag'].keys()):
        _gpu_id = jin['gpu_enable'][i % len(jin['gpu_enable'])] if len(jin['gpu_enable']) != 0 else -1
        pve.append(
            multiprocessing.Process(
                target=train_task, 
                args=(
                    tag, 
                    f"{jin['data_path']}/{tag}.txt",
                    f"{jin['log_path']}/{tag}.log",
                    jin['fig_path'],
                    _gpu_id,
                    jin['data_tag'][tag],
                    jin['train_param'],
                    jin['data_construct_param'],
                )
            )
        )

    [p.start() for p in pve]
    [p.join() for p in pve]
