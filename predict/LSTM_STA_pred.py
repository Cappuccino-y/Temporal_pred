import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from LSTM_Model import STA_LSTM
import math
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.makedirs('../model', exist_ok=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
myseed = 42000
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

def windows_input(input, output, window_range, pred_range):
    num = len(input) - window_range + 1 - pred_range
    input = input.unsqueeze(0)
    output = output.unsqueeze(0)
    seq_input = []
    for i in range(num):
        seq_input.append(input[:, i:i + window_range, :])
    input = torch.cat(seq_input, dim=0)
    seq_target = []
    for j in range(num):
        seq_target.append(output[:, window_range + j - 1:window_range + j - 1 + pred_range, :])
    output = torch.cat(seq_target, dim=0)
    return input, output

window_length=100
pred_length=10
data = pd.read_csv("../data/data_final.csv", index_col=0)
feature_num=len(data.columns)-1
data = data.values
dev_split_start=round(len(data)*0.6)
dev_split_end=dev_split_start+round(len(data)*0.2)


input = torch.from_numpy(data[:,:feature_num]).float()
target = torch.from_numpy(data[:,feature_num:]).float()

# 现在对数据集进行随机切分
# split_point = round(split_ratio * data.shape[0])
# data_train = data[:split_point, :-target_size]
# data_test = data[split_point:, :-target_size]
# data_train_target = data[:split_point, -target_size:]
# data_test_target = data[split_point:, -target_size:]
# ss = StandardScaler()
# data_train = ss.fit_transform(data_train)
# data_test = ss.transform(data_test)
# feature_num = data.shape[1] - target_size

config = {
    'epoch_num': 450,
    'batch_size': 150,
    'optim_hyper': {'lr': 0.0015, 'weight_decay': 0},
    'early_stop': 60,
    'optimizer': 'Adam',
    'save_name': 'sintering_LSTM_STA.pth'
}

class dataset(Dataset):
    def __init__(self,
                 train_input,
                 train_target,
                 window_length,
                 pred_length,
                 mode='train'
                 ):
        self.mode = mode
        input_seq, output_seq = windows_input(train_input, train_target, window_length, pred_length)
        output_seq = output_seq.squeeze(-1)

        if mode == 'train':
            indices = [i for i in range(dev_split_start) ]
        elif mode == 'dev':
            indices = [i for i in range(dev_split_start,dev_split_end) ]
        elif mode=='test':
            indices=[i for i in range(dev_split_end,len(input_seq)) ]
        self.data = input_seq[indices]
        self.target = output_seq[indices]

    def __getitem__(self, index):
        return self.data[index], self.target[index]
        # if self.mode in ['train', 'dev']:
        #     return self.data[index], self.target[index]
        # else:
        #     return self.data[index]

    def __len__(self):
        return len(self.data)

dataset_train = dataset(input,target,window_length,pred_length,mode='train')
dataset_dev=dataset(input,target,window_length,pred_length,mode='dev')
dataset_test=dataset(input,target,window_length,pred_length,mode='test')
