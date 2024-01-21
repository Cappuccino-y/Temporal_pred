import torch
import torch.nn as nn
import math




class STA_LSTM(nn.Module):
    def __init__(self, feature_num, sa_hidden, ta_hidden, output_size, length,temp_node,spat_node):
        super().__init__()
        # 超参数继承
        self.input_dim = feature_num
        self.sa_hidden = sa_hidden
        self.ta_hidden = ta_hidden
        self.seq_length = length
        self.output_dim = output_size

        # 预测模型
        self.SA =nn.LSTM(input_size=feature_num,hidden_size=sa_hidden,num_layers=1,batch_first=True)
        self.fc=nn.Linear(sa_hidden,output_size)

    def forward(self, X,mode):
        hidden_seq, (_, _) = self.SA(X)
        y_pred=self.fc(hidden_seq[:,-1])
        return y_pred
