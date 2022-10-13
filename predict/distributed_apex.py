import os
import tempfile
import math
import torch.distributed as dist
import numpy as np
import pandas as pd
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from apex.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from LSTM_Model import STA_LSTM
from apex import amp
import random

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '42594'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
myseed = 42000


os.makedirs('model', exist_ok=True)
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
    'epoch_num': 500,
    'batch_size': 150,
    'optim_hyper': {'lr': 0.0015, 'weight_decay': 0},
    'early_stop': 60,
    'optimizer': 'Adam',
    'save_name': 'sintering_model.pth'
}
loss_function = nn.MSELoss()
records = []

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


dataset_train = dataset(input,target,window_length,pred_length,mode='train')
dataset_dev=dataset(input,target,window_length,pred_length,mode='dev')
dataset_test=dataset(input,target,window_length,pred_length,mode='test')
test_dataloader = DataLoader(dataset=dataset_test,
                             batch_size=config['batch_size'],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             num_workers=0)


def set_up(rank, world_size):
    torch.cuda.set_device(rank)
    # 初始化某号进程组
    # 注：第一个参数选择后端，nccl后端是单机多卡情况下的推荐，比gloo快很多。
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def train_pro(train_dataloader, dev_dataloader, model, device, loss_fn, rank):
    early_point = 0
    min_loss = 10000
    # create model and move it to GPU with id rank
    model = model.cuda(rank)
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hyper'])
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level='O1')
    ddp_model = DDP(model)
    for epoch in range(config['epoch_num']):
        ddp_model.train()
        for x, y in train_dataloader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = ddp_model(x)
            loss = torch.sqrt(loss_fn(pred, y))
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        train_loss = dev_pro(dev_dataloader, ddp_model, device, loss_fn)
        records.append(train_loss)
        if (train_loss < min_loss):
            min_loss = train_loss
            if (rank == 0):
                print('Saving model (epoch = {:4d}, loss = {:.4f})'
                      .format(epoch + 1, min_loss))
                torch.save(ddp_model.state_dict(), 'model/{}'.format(config['save_name']))
            early_point = 0
        else:
            early_point += 1
        if early_point >= config['early_stop']:
            print("stop")
            if (rank == 0):
                loss_dev = pd.DataFrame(records)
                loss_dev.to_csv("loss_dev.csv")
            break
    # if rank == 0:
    #     print('Finished after {}'.format(epoch + 1))
    return min_loss


def dev_pro(dev_dataloader, model, device, loss_fn):
    model.eval()
    total_loss = 0
    for x, y in dev_dataloader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            loss = loss_fn(pred, y)
        total_loss += loss.detach().cpu().item()
    total_loss = math.sqrt(total_loss / len(dev_dataloader))
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


def run_fn(rank, world_size):
    print(f"Running on rank {rank}.")
    set_up(rank, world_size)

    train_sampler = DistributedSampler(dataset_train)
    dl_train_dist = DataLoader(dataset=dataset_train,
                               batch_size=config['batch_size'],
                               sampler=train_sampler,
                               shuffle=False)
    dev_sampler = DistributedSampler(dataset_dev)
    dl_dev_dist = DataLoader(dataset=dataset_dev,
                             batch_size=config['batch_size'],
                             sampler=dev_sampler,
                             shuffle=False)

    model = STA_LSTM(feature_num=feature_num, sa_hidden=40, ta_hidden=40,
                     output_size=pred_length, length=window_length)

    RMSE = train_pro(
        dl_train_dist,
        dl_dev_dist,
        model,
        device='cuda',
        loss_fn=loss_function,
        rank=rank
    )


def main(run_fn, world_size):
    mp.spawn(run_fn, args=(world_size,), nprocs=world_size, join=True)




def set_seed(seed):
    # 必须禁用模型初始化中的任何随机性。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus

    set_seed(myseed)
    main(run_fn, world_size)
    model_1 = STA_LSTM(feature_num=feature_num, sa_hidden=40, ta_hidden=40,
                       output_size=pred_length, length=window_length).cuda()
    os.makedirs('../distributed_model', exist_ok=True)
    ckpt = torch.load('../distributed_model/{}'.format(config['save_name']), map_location='cuda')
    model_1.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    preds, loss = test(test_dataloader, model_1, "cuda")

    loss_test = pd.DataFrame(loss.numpy())
    preds=pd.DataFrame()

    # _, y_true = windows_input(da, data_test_target, lookback, time_range)
    # y_true = y_true.squeeze().numpy()
    # r2_loss = r2_score(y_true, preds, multioutput='raw_values')
    #
    # r2_loss_test = pd.DataFrame(r2_loss)
    # loss_test.to_csv("loss_test.csv")
    # r2_loss_test.to_csv("r2_loss_test.csv")
