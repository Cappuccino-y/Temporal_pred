import numpy as np
import pandas as pd
import torch

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
        seq_target.append(output[:, window_range + j :window_range + j  + pred_range, :])
    output = torch.cat(seq_target, dim=0)
    return input, output

window_length=60
pred_length=7
data = pd.read_csv("../data_optimal/data_time_reconstruct.csv", index_col=0)
data_append=pd.read_csv("../data_optimal/data_append.csv", index_col=0)
feature_num_append=data_append.shape[1]

data=pd.concat([data_append,data],axis=1)
feature_num=len(data.columns)
target_num=1
data = data.values
input = data[:,:feature_num]

target = torch.from_numpy(data[:,data.shape[1]-target_num:]).float()
point=round(len(input)*0.9)
input_train_dev=input[:point]
input=torch.from_numpy(input).float()
target_train_dev=target[:point]
input_test=input[point:]
target_test=target[point:]
_, y_true = windows_input(input_test, target_test, window_length, pred_length)
y_true=y_true.squeeze()
y_true=pd.DataFrame(y_true.numpy())
y_true.to_csv('./true_res.csv')
