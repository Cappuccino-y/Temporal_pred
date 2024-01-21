import torch
import torch.nn as nn
import math


class TemporalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, temp_node,attention_node):
        super(TemporalLSTM, self).__init__()

        # 超参数继承
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.former_seq_len = seq_len
        self.output_dim=output_dim
        self.attention_node=attention_node


        # 注意力的参数
        self.Wa = nn.Parameter(torch.Tensor(input_dim, output_dim, temp_node), requires_grad=True)
        self.Ua = nn.Parameter(torch.Tensor(hidden_dim,  output_dim, temp_node), requires_grad=True)
        self.ba = nn.Parameter(torch.Tensor( output_dim, temp_node), requires_grad=True)
        self.Va = nn.Parameter(torch.Tensor(temp_node, output_dim, 1), requires_grad=True)
        self.Softmax = nn.Softmax(dim=1)

        self.Wa_sa = nn.Parameter(torch.Tensor(input_dim+1, output_dim, attention_node), requires_grad=True)
        self.Ua_sa = nn.Parameter(torch.Tensor(1, output_dim, attention_node), requires_grad=True)
        self.ba_sa = nn.Parameter(torch.Tensor(output_dim, attention_node), requires_grad=True)
        self.Va_sa = nn.Parameter(torch.Tensor(attention_node, output_dim, 1), requires_grad=True)

        # LSTM参数
        self.W = nn.Parameter(torch.Tensor(input_dim+1, hidden_dim * 4), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4), requires_grad=True)
        self.fc = nn.Linear(hidden_dim, 1, bias=True)
        self.dropout=nn.Dropout(p=0.5)

        self.h_t= nn.Parameter()
        self.LSTM_c_t = nn.Parameter()

        # 权重初始化
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.normal_(0, stdv)
            # weight.data.uniform_(-stdv, stdv)

    def forward(self,H, target_init):
        HS = self.hidden_dim

        # 参数命名
        h = H
        # 参数取得便于后续操作
        batch_size, _, input_dim = H.size()
        y_t_1=[target_init]

        # 隐藏序列
        hidden_seq = []

        # 初始状态
        # s_t = torch.zeros(batch_size, self.hidden_dim).to(h.device)
        LSTM_h_t = torch.zeros(batch_size, self.hidden_dim).to(h.device)
        LSTM_c_t = torch.zeros(batch_size, self.hidden_dim).to(h.device)

        # 打循环开始
        t = 0
        seq_len=self.output_dim
        # 注意力机制的计算
        while t < seq_len:
            # h_t = h
            # 计算注意力(第二个维度对应了是时间序列长度)
            beta_t = torch.tanh(
                h @ self.Wa[:,t] + (LSTM_c_t @ self.Ua[:,t]).unsqueeze(1).repeat(1, self.former_seq_len, 1) + self.ba[t]) @ self.Va[:, t]

            # softmax过一次
            beta_t = beta_t.squeeze()
            beta_t = self.Softmax(beta_t / math.sqrt(HS))
            # 扩充对齐inpupt_dim维度(重复之后直接做哈达玛积运算)
            beta_t = beta_t.unsqueeze(2)
            beta_t = beta_t.repeat(1, 1, input_dim)

            # 合并掉时间序列的维度(全序列)
            h_t = torch.sum(input=beta_t * h, dim=1)
            h_t=torch.cat([h_t,y_t_1[t]],dim=1)

            # 计算注意力
            a_t = torch.tanh(h_t.unsqueeze(dim=2).repeat(1, 1, self.attention_node) * self.Wa_sa[:, t, :] + (y_t_1[t] @ self.Ua_sa[:, t]).unsqueeze(1).repeat(1, self.input_dim+1, 1)+ self.ba_sa[t,:]) @ self.Va_sa[:,t]
            # softmax归一化
            alpha_t = self.Softmax(a_t / 1)

            alpha_t= alpha_t.squeeze(2)
            h_t = alpha_t * h_t
            # LSTM门值的计算(y加进去算)
            gates = h_t @ self.W + LSTM_h_t @ self.U + self.bias

            i_t = torch.sigmoid(gates[:, :HS])
            f_t = torch.sigmoid(gates[:, HS:HS * 2])
            g_t = torch.tanh(gates[:, HS * 2:HS * 3])
            o_t = torch.sigmoid(gates[:, HS * 3:])

            # 隐藏层状态的计算
            LSTM_c_t = f_t * LSTM_c_t + i_t * g_t
            LSTM_h_t = o_t * torch.tanh(LSTM_c_t)
            hidden_seq.append(LSTM_h_t.unsqueeze(0))
            y_t_1.append( self.fc(self.dropout(LSTM_h_t)))

            # 时刻加一
            t = t + 1
        # 隐藏状态的计算
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        y_t_1=torch.cat(y_t_1[1:],dim=1)

        return y_t_1, hidden_seq


class SpatialLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim,seq_len):
        super(SpatialLSTM, self).__init__()

        # 超参数继承
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.attention_node = attention_node

        # 向量化
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4), requires_grad=True)

        # 注意力的参数
        # self.Wa = nn.Parameter(torch.Tensor(input_dim, seq_len, attention_node), requires_grad=True)
        # self.Ua = nn.Parameter(torch.Tensor(hidden_dim, seq_len, attention_node), requires_grad=True)
        # self.ba = nn.Parameter(torch.Tensor(seq_len, attention_node), requires_grad=True)
        # self.Va = nn.Parameter(torch.Tensor(attention_node, seq_len, 1), requires_grad=True)
        # self.Softmax = nn.Softmax(dim=1)

        # 权重初始化
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.normal_(0, stdv)
            # weight.data.uniform_(-stdv, stdv)

    def forward(self, X):

        # 参数取得便于后续操作
        batch_size, seq_len, _ = X.size()

        # 参数命名
        x = X

        # 隐藏序列
        hidden_seq = []

        # 初始值计算
        h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        # 序列长度的计算
        HS = self.hidden_dim

        # 打循环开始
        t = 0
        # LSTM的计算
        while t < seq_len:
            # 取出当前的值
            x_t = x[:, t, :]

            # # 计算注意力
            # a_t = torch.tanh(x_t.unsqueeze(dim=2).repeat(1, 1, self.attention_node) * self.Wa[:, t, :] + (c_t @ self.Ua[:, t]).unsqueeze(1).repeat(1, self.input_dim, 1)+ self.ba[t,:]) @ self.Va[:,t]
            # # softmax归一化
            # alpha_t = self.Softmax(a_t / 1)
            #
            # alpha_t= alpha_t.squeeze(2)
            # 加权
            # x_t = alpha_t * x_t

            # 计算门值
            gates = x_t @ self.W + h_t @ self.U + self.bias

            i_t = torch.sigmoid(gates[:, :HS])
            f_t = torch.sigmoid(gates[:, HS:HS * 2])
            g_t = torch.tanh(gates[:, HS * 2:HS * 3])
            o_t = torch.sigmoid(gates[:, HS * 3:])

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

            t = t + 1
        # 隐藏状态的计算
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)
    #alpha_t


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
        self.conv
        self.SA = SpatialLSTM(input_dim=feature_num, hidden_dim=sa_hidden,seq_len=length)
        self.TA = TemporalLSTM(input_dim=sa_hidden, hidden_dim=ta_hidden, seq_len=length, output_dim=output_size,temp_node=temp_node,attention_node=spat_node)

    def forward(self, X):
        hidden_seq, (_, _) = self.SA(X)
        y_pred, _ = self.TA(hidden_seq,X[:,-1,-1:])
        return y_pred
