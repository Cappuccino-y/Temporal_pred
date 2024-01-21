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
from LSTM_Model import STA_LSTM
import math
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.makedirs('../model', exist_ok=True)

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



window_length=50
pred_length=7
data = pd.read_csv("../data/data_final.csv", index_col=0)
feature_num=len(data.columns)
target_num=1
sa_hidden=window_length*2
ta_hidden=window_length
spat_node=feature_num*2
temp_node=window_length
kernel_size=2
out_channel=feature_num
loss_T=pred_length
data = data.values
ss = StandardScaler()

input = data[:,:feature_num]
input= ss.fit_transform(input)
#权重分配
# input=np.concatenate([input[:,:-2]/38,input[:,-2:-1]*7/38,input[:,-1:]*20/38],axis=1)
input=torch.from_numpy(input).float()


target = torch.from_numpy(data[:,data.shape[1]-target_num:]).float()
point=round(len(input)*0.9)
input_train_dev=input[:point]
target_train_dev=target[:point]
input_test=input[point:]
target_test=target[point:]

config = {
    'epoch_num': 200,
    'batch_size': 64,
    'optim_hyper': {'lr': 0.001, 'weight_decay': 0},
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

        dev_split_start = round(len(input_seq) * 0.2)
        if mode == 'train':
            indices = [i for i in range(dev_split_start) ]
        elif mode == 'dev':
            indices = [i for i in range(dev_split_start,len(input_seq)) ]
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

def train_pro(zipped_list, model, device, loss_fn):
    early_point = 0
    loss =10000
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hyper'])
    count = 0
    loss_temp = torch.ones([2, pred_length])
    for epoch in range(config['epoch_num']):
        model.train()
        for index,(train_dataloader,dev_dataloader) in enumerate(zipped_list):
            train_loss=0
            for x, y in train_dataloader:
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                pred = model(x)
                # l=torch.sqrt(torch.sum((pred - y) ** 2*(torch.tensor([1/(y.shape[1]-N) for N in range(y.shape[1])])).to(device), dim=1)/y.shape[1])
                if(count>1):
                    change_rate=loss_temp[1,:]/loss_temp[0,:]
                    weight=torch.softmax(change_rate/loss_T,dim=0).to(device)
                    l = torch.sqrt(torch.sum((pred - y) ** 2/y.shape[0], dim=0))
                else:
                    weight=(torch.ones(y.shape[1])/y.shape[1]).to(device)
                    l = torch.sqrt(torch.sum((pred - y) ** 2/y.shape[0], dim=0))
                loss_temp[0,:]=loss_temp[1,:]
                loss_temp[1,:]=l.detach()
                l=torch.sum(l*weight,dim=0)
                l.backward()
                optimizer.step()
                count+=1
            train_loss += dev_pro(dev_dataloader, model, device, loss_fn)/fold_num
        if train_loss < loss :
            loss = train_loss
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                  .format(epoch + 1, loss))
            torch.save(model.state_dict(), '../model/{}'.format(config['save_name']))
            early_point = 0
        else:
            early_point += 1
        if early_point >= config['early_stop']:
        #         zipped_list.remove((train_dataloader,dev_dataloader))
        # if not zipped_list:
            break
    print('Finished after {}'.format(epoch + 1))
    return loss, loss_temp

def dev_pro(dev_dataloader,model,device,loss_fn):
    model.eval()
    total_loss=0
    for x,y in dev_dataloader:
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            pred=model(x)
            loss = torch.sum((pred - y) ** 2, dim=0)
        total_loss += loss.detach().cpu()
    total_loss = math.sqrt(torch.sum(total_loss )/ len(dev_dataloader.dataset)) / pred_length
    return total_loss

def test(tt_set, model, device):
    model.eval()  # set model to evalutation mode
    total_loss = torch.zeros(pred_length)
    preds = []
    for x, y in tt_set:  # iterate through the dataloader
        x = x.to(device)  # move data to device (cpu/cuda)
        y = y.to(device)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            preds.append(pred.detach().cpu())  # collect prediction
            loss = torch.sum((pred - y) ** 2, dim=0)
        total_loss += loss.detach().cpu()
    total_loss = torch.sqrt(total_loss / len(tt_set.dataset))
    preds = torch.cat(preds, dim=0).numpy()  # concatenate all predictions and convert to a numpy array
    return preds, total_loss

train_list=[]
dev_list=[]

fold_num=10
for fold in range(fold_num):
    start=round(fold*len(input_train_dev)/10)
    end= round((fold+1)*len(input_train_dev)/10) if (fold+1)*len(input_train_dev)/10<=len(input_train_dev) else len(input_train_dev)
    input_k=input_train_dev[start:end]
    target_k=target_train_dev[ start:end ]
    dataset_train = dataset(input_k,target_k,window_length,pred_length,mode='train')
    dataset_dev=dataset(input_k,target_k,window_length,pred_length,mode='dev')
    dataloader_train=DataLoader(dataset_train,batch_size=config['batch_size'],shuffle=True,pin_memory=True)
    dataloader_dev=DataLoader(dataset_dev,batch_size=config['batch_size'],shuffle=False,pin_memory=True)
    train_list.append(dataloader_train)
    dev_list.append(dataloader_dev)

zipped_list=list(zip(train_list,dev_list))

dataset_test=dataset(input_test,target_test,window_length,pred_length,mode='test')
dataloader_test=DataLoader(dataset_test,batch_size=config['batch_size'],shuffle=False)


model = STA_LSTM(feature_num=feature_num,kernel_size=kernel_size,out_channel=out_channel,sa_hidden=sa_hidden,ta_hidden=ta_hidden,output_size=pred_length,length=window_length,spat_node=spat_node,temp_node=temp_node)
model.to(get_device())
loss23,temp=train_pro(zipped_list,model,get_device(),loss_function)

del model

model = STA_LSTM(feature_num=feature_num,kernel_size=kernel_size,out_channel=out_channel,sa_hidden=sa_hidden,ta_hidden=ta_hidden,output_size=pred_length,length=window_length,spat_node=spat_node,temp_node=temp_node)
model.load_state_dict(torch.load('../model/{}'.format(config['save_name'])))
model.to(get_device())
preds,loss = test(dataloader_test, model, get_device())

preds=pd.DataFrame(preds)
loss=pd.DataFrame(loss.numpy())

_, y_true = windows_input(input_test, target_test, window_length, pred_length)
y_true = y_true.squeeze().numpy()
r2_loss = r2_score(y_true, preds, multioutput='raw_values')

r2_loss_test = pd.DataFrame(r2_loss)
r2_loss_test.to_csv("../res/r2_loss_test.csv")

preds.to_csv("../res/preds.csv")
loss.to_csv("../res/loss.csv")

test_true=pd.DataFrame(target_test[window_length].numpy())
test_true.to_csv("../res/pred_true.csv")