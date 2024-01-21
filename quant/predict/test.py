import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from models.LSTM_Model2 import STA_LSTM
from pcgrad import PCGrad
import math
from manager_torch import GPUManager
gpu_chooser=GPUManager()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_chooser.auto_choice())
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
window_length=30
pred_length=1
data_info = pd.read_csv("../data_train/data.csv", index_col=0)
data_append=pd.read_csv("../data_train/data_append.csv",index_col=0)
feature_num_append=data_append.shape[1]

data_info=pd.concat([data_append,data_info],axis=1)
feature_num=len(data_info.columns)
target_num=1
sa_hidden=70
ta_hidden=70
temp_node=70
kernel_size=2
loss_T=5

model = STA_LSTM(feature_num=feature_num-feature_num_append,feature_num_append=feature_num_append,sa_hidden=sa_hidden,ta_hidden=ta_hidden,output_size=pred_length,length=window_length,temp_node=temp_node)
model.to(get_device())

