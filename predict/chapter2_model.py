import torch
import torch.nn as nn
import math
from torch.nn.utils import weight_norm


class TemporalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, former_seq_len, output_dim, temp_node):
        super(TemporalLSTM, self).__init__()

        # 超参数继承
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.former_seq_len = former_seq_len
        self.output_dim=output_dim


        # 注意力的参数
        self.Wa = nn.Parameter(torch.Tensor(input_dim, output_dim, temp_node), requires_grad=True)
        self.Ua = nn.Parameter(torch.Tensor(hidden_dim,  output_dim, temp_node), requires_grad=True)
        self.ba = nn.Parameter(torch.Tensor( output_dim, temp_node), requires_grad=True)
        self.Va = nn.Parameter(torch.Tensor(temp_node, output_dim, 1), requires_grad=True)
        self.Softmax = nn.Softmax(dim=1)

        # LSTM参数
        self.W = nn.Parameter(torch.Tensor(input_dim,output_dim, hidden_dim * 4), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(hidden_dim,output_dim, hidden_dim * 4), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(output_dim,hidden_dim * 4), requires_grad=True)
        self.output_fc=nn.ModuleList(nn.Linear(hidden_dim,1,bias=True) for _ in range(output_dim))

        self.W_y = nn.Parameter(torch.Tensor(1, output_dim,hidden_dim * 4), requires_grad=True)
        # self.W_forget=nn.Parameter(torch.Tensor(input_dim+hidden_dim,1),requires_grad=True)


        # 权重初始化
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.normal_(0, stdv)
            # weight.data.uniform_(-stdv, stdv)

    def forward(self, H,y_0,embedding):
        HS = self.hidden_dim

        # 参数命名
        h = H
        # 参数取得便于后续操作
        batch_size, _, input_dim = H.size()
        y_t_1=[y_0]

        # 隐藏序列
        hidden_seq = []

        # 初始状态
        # s_t = torch.zeros(batch_size, self.hidden_dim).to(h.device)
        LSTM_h_t =embedding[ :,:self.hidden_dim ]
        LSTM_c_t =embedding[:,self.hidden_dim:]

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
            # h_t=torch.cat([h_t,y_t_1[t]],dim=1)

            # LSTM门值的计算(y加进去算)
            gates = h_t @ self.W[:,t] + LSTM_h_t @ self.U[:,t] + self.bias[t]+y_t_1[t]@self.W_y[:,t]

            i_t = torch.sigmoid(gates[:, :HS])
            f_t = torch.sigmoid(gates[:, HS:HS * 2])
            g_t = torch.tanh(gates[:, HS * 2:HS * 3])
            o_t = torch.sigmoid(gates[:, HS * 3:])

            # 隐藏层状态的计算
            LSTM_c_t = f_t * LSTM_c_t + i_t * g_t
            LSTM_h_t = o_t * torch.tanh(LSTM_c_t)
            hidden_seq.append(LSTM_h_t.unsqueeze(0))
            y_t_1.append( self.output_fc[t](LSTM_h_t))

            # 时刻加一
            t = t + 1
        # 隐藏状态的计算
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        y_t_1=torch.cat(y_t_1[1:],dim=1)

        return y_t_1, hidden_seq


class SpatialLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SpatialLSTM, self).__init__()

        # 超参数继承
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 向量化
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4), requires_grad=True)

        # # 注意力的参数
        # self.Wa = nn.Parameter(torch.Tensor(input_dim, input_dim), requires_grad=True)
        # self.Ua = nn.Parameter(torch.Tensor(hidden_dim , input_dim), requires_grad=True)
        # self.ba = nn.Parameter(torch.Tensor(input_dim), requires_grad=True)
        # self.Va = nn.Parameter(torch.Tensor(input_dim, input_dim), requires_grad=True)
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
            # a_t = torch.tanh(x_t @ self.Wa + c_t @ self.Ua + self.ba) @ self.Va
            #
            # # softmax归一化
            # alpha_t = self.Softmax(a_t / 1)
            #
            # # 加权
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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)# Squeeze操作的定义
        self.fc = nn.Sequential(# Excitation操作的定义
            nn.Linear(channel, channel // reduction, bias=False),# 压缩
            nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),# 恢复
            nn.Sigmoid()# 定义归一化操作
        )

    def forward(self, x):
        b, c, _ = x.size()# 得到H和W的维度，在这两个维度上进行全局池化
        y = self.avg_pool(x).view(b, c)# Squeeze操作的实现
        y = self.fc(y).view(b, c, 1)# Excitation操作的实现
        # 将y扩展到x相同大小的维度后进行赋权
        return x * y.expand_as(x)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.PReLU()
        self.init_weights()
        self.attention_spatial=SELayer(n_outputs)

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        out=self.attention_spatial(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.3):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class STA_LSTM(nn.Module):
    def __init__(self, feature_num,feature_num_append,sa_hidden, ta_hidden, output_size, length,temp_node):
        super().__init__()
        # 超参数继承
        self.input_dim =40
        self.sa_hidden = sa_hidden
        self.ta_hidden = ta_hidden
        self.seq_length = length
        self.output_dim = output_size
        self.feature_num_append=feature_num_append


        # self.net1=nn.Sequential(
        # nn.Conv1d(in_channels=feature_num, kernel_size=3, out_channels=feature_num,padding=1),
        # nn.Conv1d(in_channels=feature_num*2, kernel_size=3, out_channels=feature_num,padding=0),
        # nn.Conv1d(in_channels=feature_num, kernel_size=3, out_channels=round(feature_num/2)),
        # nn.ReLU(),
        # nn.BatchNorm1d(num_features=(feature_num))
        # )

        self.net1=TemporalConvNet(feature_num,[self.input_dim,self.input_dim*2,self.input_dim*4,self.input_dim*4,self.input_dim*2,self.input_dim],kernel_size=2)
        # self.net2=nn.Sequential(
        # nn.LayerNorm(normalized_shape=(length , feature_num)),
        # nn.Dropout(p=0.1)
        # )
        # 预测模型

        # self.SA = SpatialLSTM(input_dim=(feature_num), hidden_dim=sa_hidden)
        self.lstm_enbedding=nn.LSTM(input_size=self.input_dim,hidden_size=sa_hidden,num_layers=1,batch_first=True)
        self.feature_change=nn.Sequential(nn.Linear(sa_hidden,2*sa_hidden),nn.PReLU(),nn.Linear(2*sa_hidden,4*ta_hidden),nn.PReLU(),nn.Linear(4*ta_hidden,2*ta_hidden),nn.PReLU())
        self.TA = TemporalLSTM(input_dim=self.input_dim, hidden_dim=ta_hidden, former_seq_len=length, output_dim=output_size, temp_node=temp_node)

        # self.init_orthogonal()

    def init_orthogonal(self):
        for name,var in self.lstm_enbedding.state_dict(keep_vars=True).items():
            if (name.find('weight')>-1):
                nn.init.orthogonal_(var)



    def forward(self, X):
        # X_embedding=X[:,:,:self.feature_num_append]
        # X=X[:,:,self.feature_num_append:]
        remap=self.net1(X.transpose(1,2).contiguous())
        remap=remap.transpose(1,2).contiguous();

        embedding,(_,_)=self.lstm_enbedding(remap)
        embedding_init=self.feature_change(embedding[:,-1])
        # remap=self.net2(remap)
        # hidden_seq, (h_t_0, c_t_0) = self.SA(remap)
        y_pred, _ = self.TA(remap,X[:,-1,-1:],embedding_init)
        return y_pred