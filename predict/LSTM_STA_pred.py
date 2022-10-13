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
        seq_target.append(output[:, window_range + j - 1:window_range + j - 1 + pred_range, :])
    output = torch.cat(seq_target, dim=0)
    return input, output

window_length=100
pred_length=10
data = pd.read_csv("../data/data_final.csv", index_col=0)
feature_num=len(data.columns)-1
data = data.values


input = torch.from_numpy(data[:,:feature_num]).float()
target = torch.from_numpy(data[:,feature_num:]).float()

config = {
    'epoch_num': 450,
    'batch_size': 150,
    'optim_hyper': {'lr': 0.0015, 'weight_decay': 0},
    'early_stop': 60,
    'optimizer': 'Adam',
    'save_name': 'sintering_model.pth'
}
loss_function = nn.MSELoss()


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

        dev_split_start = round(len(input_seq) * 0.6)
        dev_split_end = dev_split_start + round(len(input_seq) * 0.2)
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

def train_pro(train_dataloader, dev_dataloader, model, device, loss_fn):
    early_point = 0
    min_loss = 10000
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hyper'])
    for epoch in range(config['epoch_num']):
        model.train()
        for x, y in train_dataloader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            l = torch.sqrt(loss_fn(pred, y))
            l.backward()
            optimizer.step()
        train_loss = dev_pro(dev_dataloader, model, device, loss_fn)
        if train_loss < min_loss or (epoch + 1) % 10 == 0:
            min_loss = train_loss
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                  .format(epoch + 1, min_loss))
            torch.save(model.state_dict(), '../model/{}'.format(config['save_name']))
            early_point = 0
        else:
            early_point += 1
        if early_point >= config['early_stop']:
            break
    print('Finished after {}'.format(epoch + 1))
    return min_loss

def dev_pro(dev_dataloader,model,device,loss_fn):
    model.eval()
    total_loss=0
    for x,y in dev_dataloader:
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            pred=model(x)
            loss=loss_fn(pred,y)
        total_loss+=loss.detach().cpu().item()
    total_loss=math.sqrt(total_loss/len(dev_dataloader))
    return total_loss

def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds

dataset_train = dataset(input,target,window_length,pred_length,mode='train')
dataset_dev=dataset(input,target,window_length,pred_length,mode='dev')
dataset_test=dataset(input,target,window_length,pred_length,mode='test')
dataloader_train=DataLoader(dataset_train,batch_size=config['batch_size'],shuffle=True,pin_memory=True)
dataloader_dev=DataLoader(dataset_dev,batch_size=config['batch_size'],shuffle=False,pin_memory=True)
dataloader_test=DataLoader(dataset_test,batch_size=config['batch_size'],shuffle=False)

model = STA_LSTM(feature_num=feature_num,sa_hidden=window_length*2,ta_hidden=window_length,output_size=pred_length,length=window_length)
model.to(get_device())
train_pro(dataloader_train,dataloader_dev,model,get_device(),loss_function)

del model

model = STA_LSTM(feature_num=feature_num,sa_hidden=window_length*2,ta_hidden=window_length,output_size=pred_length,length=window_length)
model.load_state_dict(torch.load('../model/{}'.format(config['save_name'])))
model.to(get_device())
preds = test(dataloader_test, model, get_device())