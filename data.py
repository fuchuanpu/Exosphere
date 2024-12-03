import os
import random
import math
from typing import List, Tuple

import torch

from common import *

T_MAX = 100
ETH_MTU = 1500

@time_log
def load_data(data_tag:str, data_target:str, 
                shuffle_seed=None, fig_path=None, 
                train_ratio=0.80, segment_len=1000, time_base=1e-6,
            ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor,
                        List[List[int]], List[List[int]], List[List[int]], List[List[int]]]:
    
    if not os.path.exists(data_target):
        exit(-1)

    logging.info(f'Read dataset from {data_target}')
    tim_vec, len_vec, label_vec = [], [], []
    with open(data_target, 'r') as f:
        for ll in f.readlines():
            [a, b, c] = list(map(float, ll.split()))
            tim_vec.append(math.floor(a * (1 / time_base)))
            len_vec.append(min(b, ETH_MTU) / ETH_MTU)
            label_vec.append(int(c))

    iat_vec = [0]
    start = tim_vec[0]
    for i in range(1, len(tim_vec)):
        iat_vec.append(min(max(tim_vec[i] - start, 0), T_MAX))
        start = max(start, tim_vec[i])

    feature_vec = []
    real_label_vec = []
    num_packet_vec = []
    num_attack_vec = []
    index = 0
    while index < len(iat_vec):
        time_var = iat_vec[index]
        feature_vec.append(-time_var/T_MAX)
        real_label_vec.append(0)
        num_packet_vec.append(0)
        num_attack_vec.append(0)

        feature_vec.append(len_vec[index])
        num_packet_vec.append(1)
        real_label_vec.append(label_vec[index])
        num_attack_vec.append(int(label_vec[index]))

        index += 1

    seg_feature_vec, seg_label_vec = [], []
    seg_num_vec, seg_attack_num_vec = [], []
    for i in range(0, len(feature_vec), segment_len):
        if i + segment_len >= len(feature_vec):
            continue
        seg_feature_vec.append(feature_vec[i:i + segment_len])
        seg_label_vec.append(real_label_vec[i:i + segment_len])
        seg_num_vec.append(num_packet_vec[i:i + segment_len])
        seg_attack_num_vec.append(num_attack_vec[i:i + segment_len])


    if shuffle_seed is not None:
        random.seed(shuffle_seed)
        _shu_ls = list(zip(seg_feature_vec, seg_label_vec, seg_num_vec, seg_attack_num_vec))
        random.shuffle(_shu_ls)
        seg_feature_vec, seg_label_vec, seg_num_vec, seg_attack_num_vec = \
            [x[0] for x in _shu_ls], [x[1] for x in _shu_ls], [x[2] for x in _shu_ls], [x[3] for x in _shu_ls]

    train_test_line = int(train_ratio * len(seg_feature_vec))
    train_data = seg_feature_vec[:train_test_line]
    train_label = seg_label_vec[:train_test_line]
    test_data = seg_feature_vec[train_test_line:]
    test_label = seg_label_vec[train_test_line:]

    train_num = seg_num_vec[:train_test_line]
    test_num = seg_num_vec[train_test_line:]
    train_atc_num = seg_attack_num_vec[:train_test_line]
    test_atc_num = seg_attack_num_vec[train_test_line:]

    train_data = torch.FloatTensor(train_data).unsqueeze(1)
    train_label = torch.FloatTensor(train_label).unsqueeze(1)
    test_data = torch.FloatTensor(test_data).unsqueeze(1)
    test_label = torch.FloatTensor(test_label).unsqueeze(1)

    logging.info(f'[{data_tag}] Attack Frames: {label_vec.count(True)}, Benign Frames: {len(label_vec) - label_vec.count(True)}.')
    logging.info(f'[{data_tag}] Train Records: {train_label.size(0)}, Test Records: {test_label.size(0)}')

    return train_data, train_label, test_data, test_label, train_num, test_num, train_atc_num, test_atc_num
