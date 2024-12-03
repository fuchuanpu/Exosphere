from typing import List
import torch
import math

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, \
    recall_score, accuracy_score, matthews_corrcoef, fbeta_score

from common import *
from loss import *
from model import *



mtx = List[List[int]]
@time_log
def train_test(data_tag:str, log_path:str, fig_path:str,
                trainD:torch.FloatTensor, trainL:torch.FloatTensor, testD:torch.FloatTensor, testL:torch.FloatTensor,
                trainN:mtx, testN:mtx, trainA:mtx, testA:mtx,
                gpu_id:int, waterline:float, lr=0.001, batch_size=4, num_epoch=15):
    
    logging.info(f'[{data_tag}] is started.')

    fout = open(log_path, 'w', buffering=1)
    exosphere = Exosphere(in_ch=1, out_ch=1)
    
    opt = torch.optim.Adam(exosphere.parameters(), lr=lr)

    train_on_gpu = torch.cuda.is_available()
    print(f'[{data_tag}] Use GPU: {gpu_id}.' if train_on_gpu else f'[{data_tag}] Use CPU.', 
            file=fout, flush=True)
    device = torch.device(f'cuda:{gpu_id}' if train_on_gpu else 'cpu')
    exosphere.to(device)
    trainD.to(device)
    trainL.to(device)
    testD.to(device)
    testL.to(device)

    print("Total Parameters:", sum([p.nelement() for p in exosphere.parameters()]), 
        file=fout, flush=True)
    
    for e in range(num_epoch):
        train_loss = 0.0
        test_loss = 0.0

        exosphere.train()
        num_train = 0
        for i in range(0, trainD.size(0), batch_size):
            if i + batch_size >= trainD.size(0):
                continue

            x = trainD[i:i + batch_size].to(device)
            y = trainL[i:i + batch_size].to(device)
            num_train += batch_size

            opt.zero_grad()
            y_pred = exosphere(x)
            lossT = calc_loss(y_pred, y)

            train_loss += lossT.item() * x.size(0)
            lossT.backward()
            opt.step()

        exosphere.eval()
        torch.no_grad()

        pred_res = []
        label_res = []
        num_res = []
        num_atc_res = []

        num_test = 0
        sum_test_time = 0

        test_batch_size = batch_size * 20
        for i in range(0, testD.size(0), test_batch_size):
            if i + test_batch_size >= testD.size(0):
                continue

            x = testD[i:i + test_batch_size].to(device)
            y = testL[i:i + test_batch_size].to(device)
            num_test += test_batch_size

            start_test = time.time()
            y_pred = exosphere(x)
            end_test = time.time()
            sum_test_time += end_test - start_test
            lossL = calc_loss(y_pred, y)

            test_loss += lossL.item() * x.size(0)
            pred_res.extend(y_pred.view(-1).tolist())
            label_res.extend(y.view(-1).tolist())
            
            for x in range(i, i + test_batch_size):
                num_res.extend(testN[x])
                num_atc_res.extend(testA[x])

        train_loss /= num_train
        test_loss /= num_test

        assert(len(num_atc_res) == len(label_res))
        assert(len(num_res) == len(pred_res))

        true_label, true_pred = [], []
        for i in range(len(label_res)):
            if num_res[i] == 0:
                continue
            else:
                true_pred.extend([pred_res[i]] * num_res[i])
                true_label.extend([1] * num_atc_res[i] + [0] * (num_res[i] - num_atc_res[i]))
        
        fpr, tpr, _ = roc_curve(true_label, true_pred)
        roc_auc = auc(fpr, tpr)
        judge = [1 if sc > waterline else 0 for sc in true_pred]
        f1 = f1_score(true_label, judge, average='macro')
        f2 = fbeta_score(true_label, judge, average='macro', beta=2)
        per = precision_score(true_label, judge, average='macro')
        rec = recall_score(true_label, judge, average='macro')
        acc = accuracy_score(true_label, judge)
        mcc = matthews_corrcoef(true_label, judge)
        fp_v, tp_v, _ = roc_curve(true_label, judge)
        if len(tp_v) != 3 or len(fp_v) != 3:
            logging.warn('Incorrect value for metrics.')
            continue

        def cal_eer(fpr, tpr):
            deta = 1
            err = 0
            for a, b in zip(fpr, tpr):
                d = math.fabs((1 - a) - b)
                if d < deta:
                    deta = d
                    err = a
            return err

        eer = cal_eer(fpr, tpr)

        packet_per_frame = testD.size(2)
        print(f'Epoch: {e:2d}, train loss: {train_loss:7.4f}, test loss: {test_loss:7.4f}, '
        f'AUC: {roc_auc:7.4f}, F1: {f1:7.4f}, Percision: {per:7.4f}, Recall: {rec:7.4f}, F2: {f2:7.4f}, '
        f'FPR: {fp_v[1]:7.4f}, TPR: {tp_v[1]:7.4f}, EER: {eer:7.4f}, MCC: {mcc:7.4f}, ACC: {acc:7.4f}, '
        f'Test Time: {sum_test_time:7.4f} s, Test Speed: {(num_test*packet_per_frame)/sum_test_time:7.2f} PPS.',
        file=fout,flush=True)

        # Save the distribution of scores
        benign_score = [x[1] for x in filter(lambda x: not x[0], list(zip(true_label, true_pred)))]
        attack_score = [x[1] for x in filter(lambda x: x[0], list(zip(true_label, true_pred)))]
        
        fig = plt.figure(figsize=(10, 10 * 0.618), constrained_layout=True)
        ax = fig.subplots(1, 1)

        ax.hist(benign_score, 1000, density=True, histtype='step', cumulative=True, label='Benign', color='royalblue')
        ax.hist(attack_score, 1000, density=True, histtype='step', cumulative=True, label='Attack', color='firebrick')
        ax.vlines(waterline, 0, 1.05, lw=1, color='grey', linestyles='--', label='Waterline')
        ax.legend(loc='right')
        ax.set_xlabel('Score')
        ax.set_ylabel('CDF')
        ax.set_title(f'Detection Accuracy: {data_tag}')

        save_addr = f'{fig_path}/{data_tag}_result.png'
        fig.savefig(save_addr, dpi=600, format='png')
        plt.cla()
