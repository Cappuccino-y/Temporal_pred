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
from models.LSTM_Model import STA_LSTM
import math
from manager_torch import GPUManager
gpu_chooser=GPUManager()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_chooser.auto_choice())
os.makedirs('./save_paras', exist_ok=True)
os.makedirs('./res', exist_ok=True)

seed=42000
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

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



window_length=14
pred_length=1
data_info = pd.read_csv("../data_train/data.csv", index_col=0)
data_append=pd.read_csv("../data_train/data_append.csv",index_col=0)
feature_num_append=data_append.shape[1]

data_info=pd.concat([data_append,data_info],axis=1)
feature_num=len(data_info.columns)
target_num=1
sa_hidden=window_length*3
ta_hidden=window_length*3
temp_node=window_length*3
kernel_size=2
loss_T=5
data = data_info.values
ss = StandardScaler()

# data=ss.fit_transform(data)
input = data[:,:feature_num]
input= ss.fit_transform(input)
#权重分配
# input=np.concatenate([input[:,:-2]/38,input[:,-2:-1]*7/38,input[:,-1:]*20/38],axis=1)
input=torch.from_numpy(input).float()

target = torch.from_numpy(data[:,data.shape[1]-target_num:]).float()
input_train_dev=input[:-window_length]
target_train_dev=target[:-window_length]
input_test=input[-window_length:]
target_test=target[-window_length:]
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

        dev_split_start = round(len(input_seq)*8/9 )
        # if mode == 'train':
        #     indices = [i for i in range(dev_split_start) ]
        # elif mode == 'dev':
        #     indices = [i for i in range(dev_split_start,len(input_seq)) ]
        if mode == 'train':
            indices = [i for i in range(len(input_seq)) if i %10 !=0 ]
        elif mode == 'dev':
            indices = [i for i in range(len(input_seq)) if i %10 ==0 ]
        elif mode=='test':
            indices=[i for i in range(len(input_seq)) ]
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

# iterate through the dataloader



# dataset_test=dataset(input_test,target_test,window_length,pred_length,mode='test')
# dataloader_test=DataLoader(dataset_test,batch_size=1,shuffle=False)

model = STA_LSTM(feature_num=feature_num-feature_num_append,feature_num_append=feature_num_append,sa_hidden=sa_hidden,ta_hidden=ta_hidden,output_size=pred_length,length=window_length,temp_node=temp_node)
model.load_state_dict(torch.load('../predict/save_paras/sintering_model_Chapter3_hybrid.pth'))
device=get_device()
model.to(device)
x = input_test.to(device).unsqueeze(0)  # move data to device (cpu/cuda)
y = target_test.to(device).unsqueeze(0)
with torch.no_grad():  # disable gradient calculation
    pred = model(x,mode="test", epoch=0)  # forward pass (compute output)
print(pred)



