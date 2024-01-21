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
from models.Autoformer import Model
import math
import argparse
from manager_torch import GPUManager
gpu_chooser=GPUManager()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_chooser.auto_choice())
os.makedirs('../model', exist_ok=True)
file_name=__file__.split("/")[-1].split(".")[0]
os.makedirs('../res_{}'.format(file_name), exist_ok=True)

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

def windows_input_ori(input, output, window_range, pred_range):
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

def windows_input(input, output, window_range, pred_range):
    num = len(input) - window_range + 1 - pred_range
    seq_input = []
    for i in range(num):
        seq_input.append(input[ i:i + window_range, :])
    input_af = torch.stack(seq_input, dim=0)
    seq_target = []
    for j in range(num):
        seq_target.append(input[ window_range + j-args.label_len :window_range + j  + pred_range, :])
    output_af = torch.stack(seq_target, dim=0)
    return input_af, output_af

def mark_generate(input, window_range, pred_range):
    num = len(input) - window_range + 1 - pred_range
    seq_input = []
    for i in range(num):
        seq_input.append(input[i:i + window_range, :])
    xmark = torch.stack(seq_input, dim=0)
    seq_target = []
    for j in range(num):
        seq_target.append(input[ window_range + j-args.label_len :window_range + j  + pred_range, :])
    ymark = torch.stack(seq_target, dim=0)
    return xmark, ymark

window_length=10
window_length_ori=60;
pred_length=7
data = pd.read_csv("../data/data_final.csv", index_col=0)
data_append=pd.read_csv("../data/data_append.csv",index_col=0)
df_stamp=pd.DataFrame(pd.date_range("2021-12-08T02:17:00", periods=len(data),freq='T'),columns=["date"])
df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
data_stamp = df_stamp.drop(['date'], 1).values
feature_num_append=data_append.shape[1]

# data=pd.concat([data_append,data],axis=1)
feature_num=len(data.columns)
target_num=1
sa_hidden=window_length
ta_hidden=window_length
temp_node=window_length
kernel_size=2
loss_T=5
data = data.values
ss = StandardScaler()
# data=ss.fit_transform(data)
input = data[:,:feature_num]
input= ss.fit_transform(input)
#权重分配
# input=np.concatenate([input[:,:-2]/38,input[:,-2:-1]*7/38,input[:,-1:]*20/38],axis=1)
input=torch.from_numpy(input).float()

ss2 = StandardScaler()
# data=ss.fit_transform(data)
data_stamp= ss2.fit_transform(data_stamp)
#权重分配
# input=np.concatenate([input[:,:-2]/38,input[:,-2:-1]*7/38,input[:,-1:]*20/38],axis=1)
data_stamp=torch.from_numpy(data_stamp).float()

target = torch.from_numpy(data[:,data.shape[1]-target_num:]).float()
point=round(len(input)*0.9)
input_train_dev=input[:point]
target_train_dev=target[:point]
input_test=input[point:]
target_test=target[point:]

config = {
    'epoch_num': 50,
    'batch_size': 128,
    'optim_hyper': {'lr': 0.0005, 'weight_decay': 0},
    'early_stop': 10,
    'optimizer': 'Adam',
    'save_name': 'sintering_model_{}.pth'.format(__file__.split("/")[-1].split(".")[0])
}
class args:
    '''Class for keeping track of an item in inventory.'''
    seq_len : int = window_length
    label_len : int =int( seq_len/2)

    pred_len : int =pred_length
    output_attention : bool= False
    moving_avg: int = 25
    enc_in:int=13
    dec_in:int =13
    c_out:int =13
    d_model:int = feature_num
    embed:str = 'timeF'
    freq:str='t'
    dropout:float=0.05
    factor:int =1
    n_heads: int= 8
    activation: str='gelu'
    e_layers:int =2
    d_ff:int =256
    d_layers:int =1


    def __post_init__(self):
        self.pt = int((self.P - self.Ck) / self.skip)
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
        xmark,ymark=mark_generate(data_stamp,window_length, pred_length)

        dev_split_start = round(len(input_seq)*8/9 )
        if mode == 'train':
            indices = [i for i in range(dev_split_start) ]
        elif mode == 'dev':
            indices = [i for i in range(dev_split_start,len(input_seq)) ]
        elif mode=='test':
            indices=[i for i in range(len(input_seq)) ]
        self.xmark=xmark[indices]
        self.ymark=ymark[indices]
        self.data = input_seq[indices]
        self.target = output_seq[indices]

    def __getitem__(self, index):
        return self.data[index], self.target[index],self.xmark[index],self.ymark[index]
        # if self.mode in ['train', 'dev']:
        #     return self.data[index], self.target[index]
        # else:
        #     return self.data[index]

    def __len__(self):
        return len(self.data)

def train_pro(train_dataloader,dev_dataloader, model, device, loss_fn):
    loss_list=[]
    early_point = 0
    loss =10000
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hyper'])
    count = 0
    loss_temp = torch.ones([2, pred_length])
    for epoch in range(config['epoch_num']):
        model.train()
        temp_store = torch.zeros(pred_length).to(device)
        for x,y,xmark,ymark in train_dataloader:
            optimizer.zero_grad()
            dec_inp = torch.zeros_like(y[:, -args.pred_len:]).float()
            dec_inp = torch.cat([y[:, :args.label_len], dec_inp], dim=1).float().to(device)
            x, y,xmark,ymark = x.to(device), y.to(device),xmark.to(device),ymark.to(device)
            pred = model(x,xmark,dec_inp,ymark)
            # l=torch.sqrt(torch.sum((pred - y) ** 2*(torch.tensor([1/(y.shape[1]-N) for N in range(y.shape[1])])).to(device), dim=1)/y.shape[1])
            # if(count>1):
            #     change_rate=loss_temp[1,:]/loss_temp[0,:]
            #     weight=torch.softmax(change_rate/loss_T,dim=0).to(device)
            #     l = torch.sqrt(torch.sum((pred - y) ** 2/y.shape[0], dim=0))
            # else:
            pred=pred[:,:,-1]
            y=y[:,-pred_length:,-1]
            weight=(torch.ones(y.shape[1])/y.shape[1]).to(device)
            l = torch.sqrt(torch.sum((pred - y) ** 2/y.shape[0], dim=0))
            loss_temp[0,:]=loss_temp[1,:]
            loss_temp[1,:]=l.detach()
            temp_store+=l.detach()
            l=torch.sum(l*weight,dim=0)
            l.backward()
            optimizer.step()
            count+=1
        train_loss = dev_pro(dev_dataloader, model, device, loss_fn)
        loss_list.append(temp_store.unsqueeze(0)/len(train_dataloader))
        if train_loss < loss :
            loss = train_loss
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                  .format(epoch + 1, loss))
            torch.save(model.state_dict(), '../model/{}'.format(config['save_name']))
            early_point = 0
        else:
            early_point += 1
        if early_point >= config['early_stop']:
            break
    print('Finished after {}'.format(epoch + 1))
    loss_list=torch.cat(loss_list,dim=0).detach().cpu().numpy()
    return loss_list, loss_temp

def dev_pro(dev_dataloader,model,device,loss_fn):
    model.eval()
    total_loss=0
    for x, y, xmark, ymark in dev_dataloader:
        dec_inp = torch.zeros_like(y[:, -args.pred_len:]).float()
        dec_inp = torch.cat([y[:, :args.label_len], dec_inp], dim=1).float().to(device)
        x, y, xmark, ymark = x.to(device), y.to(device), xmark.to(device), ymark.to(device)
        with torch.no_grad():
            pred = model(x, xmark, dec_inp, ymark)
            pred=pred[:,:,-1]
            y=y[:,-pred_length:,-1]
            loss = torch.sum((pred - y) ** 2, dim=0)
        total_loss += loss.detach().cpu()
    total_loss = torch.sum(torch.sqrt(total_loss / len(dev_dataloader.dataset))) / pred_length
    return total_loss

def test(tt_set, model, device):
    model.eval()  # set model to evalutation mode
    total_loss = torch.zeros(pred_length)
    preds = []
    for x, y, xmark, ymark in tt_set:
        dec_inp = torch.zeros_like(y[:, -args.pred_len:]).float()
        dec_inp = torch.cat([y[:, :args.label_len], dec_inp], dim=1).float().to(device)
        x, y, xmark, ymark = x.to(device), y.to(device), xmark.to(device), ymark.to(device)
        with torch.no_grad():
            pred = model(x, xmark, dec_inp, ymark)
            pred=pred[:,:,-1]
            y=y[:,-pred_length:,-1]
            preds.append(pred.detach().cpu())  # collect prediction
            loss = torch.sum((pred - y) ** 2, dim=0)
        total_loss += loss.detach().cpu()
    total_loss = torch.sqrt(total_loss / len(tt_set.dataset))
    preds = torch.cat(preds, dim=0).numpy()  # concatenate all predictions and convert to a numpy array
    return preds, total_loss

train_list=[]
dev_list=[]

dataset_train=dataset(input_train_dev,target_train_dev,window_length,pred_length,mode='train')
dataset_dev=dataset(input_train_dev,target_train_dev,window_length,pred_length,mode='dev')
dataset_test=dataset(input_test,target_test,window_length,pred_length,mode='test')
dataloader_train=DataLoader(dataset_train,batch_size=config['batch_size'],shuffle=True)
dataloader_dev=DataLoader(dataset_dev,batch_size=config['batch_size'],shuffle=False)
dataloader_test=DataLoader(dataset_test,batch_size=config['batch_size'],shuffle=False)





model = Model(args)
model.to(get_device())
loss_train,temp=train_pro(dataloader_train,dataloader_dev,model,get_device(),loss_function)

del model

model = Model(args)
model.load_state_dict(torch.load('../model/{}'.format(config['save_name'])))
model.to(get_device())
preds,loss2 = test(dataloader_test, model, get_device())
preds=preds[(window_length_ori-window_length):]
preds=preds*math.sqrt(ss.var_[-1])+ss.mean_[-1]

preds=pd.DataFrame(preds)
loss_train=pd.DataFrame(loss_train)

_, y_true = windows_input_ori(input_test, target_test, window_length, pred_length)
y_true = y_true.squeeze().numpy()
y_true=y_true[(window_length_ori-window_length):]


r2_loss = r2_score(y_true, preds, multioutput='raw_values')
loss =np.sqrt( mean_squared_error(y_true,preds , multioutput='raw_values'))
loss=pd.DataFrame(loss)

r2_loss_test = pd.DataFrame(r2_loss)
r2_loss_test.to_csv("../res_{}/r2_loss_test_{}.csv".format(file_name,file_name))


loss_train.to_csv("../res_{}/loss_train_{}.csv".format(file_name,file_name))

preds.to_csv("../res_{}/preds_{}.csv".format(file_name,file_name))
loss.to_csv("../res_{}/loss_{}.csv".format(file_name,file_name))



test_true=pd.DataFrame(target_test[window_length_ori:-pred_length+1].numpy())
test_true.to_csv("../res_{}/pred_true_{}.csv".format(file_name,file_name))