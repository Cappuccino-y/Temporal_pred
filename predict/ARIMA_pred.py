import math

from statsmodels.tsa import arima_model
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

import torch

def windows_input(input, output, window_range, pred_range):
    num = len(input) - window_range + 1 - pred_range
    input = input.unsqueeze(0)
    output = output.unsqueeze(0)
    seq_input = []
    for i in range(num):
        seq_input.append(input[:, i:i + window_range])
    input = torch.cat(seq_input, dim=0)
    seq_target = []
    for j in range(num):
        seq_target.append(output[:, window_range + j :window_range + j  + pred_range])
    output = torch.cat(seq_target, dim=0)
    return input, output

window=60
pred_length=7
data=pd.read_csv("../data_optimal/data_final.csv",index_col=0)
data_train= data.iloc[round(len(data) * 0.9) + window-10000:round(len(data) * 0.9) + window, -1].values
data_test=data.iloc[-200:,-1].values
data_train=pd.DataFrame(data_train)


# plot_acf(data_train)
# diff1=data_train.diff(1).dropna()
#
# print(u'差分序列的白噪声检验结果为：', acorr_ljungbox( diff1, lags=1))
# diff1.plot()
# plot_acf(diff1)
# plot_pacf(diff1)


input_win,output_win=windows_input(torch.from_numpy(data_test),torch.from_numpy(data_test),window,pred_length)


test_num=input_win.shape[0]
res_sum=np.zeros([test_num,pred_length])
for i in range(test_num):
    model = ARIMA(input_win[i].numpy(), order=(2, 1, 2)).fit(method='innovations_mle')
    res_sum[i]=model.forecast(pred_length)

r2_loss = r2_score(output_win[:test_num], res_sum, multioutput='raw_values')


r2_loss_test = pd.DataFrame(r2_loss)
r2_loss_test.to_csv("../res_ARIMA/r2_loss_test_{}.csv".format(__file__.split("\\")[-1].split(".")[0]))


loss=np.sqrt(np.sum((res_sum-output_win[:test_num].numpy())**2/len(res_sum),axis=0))
loss= pd.DataFrame(loss)
loss.to_csv("../res_ARIMA/loss_test{}.csv".format(__file__.split("\\")[-1].split(".")[0]))

res_sum=pd.DataFrame(res_sum)
res_sum.to_csv("../res_ARIMA/preds_{}.csv".format(__file__.split("\\")[-1].split(".")[0]))

# rmse=math.sqrt(mean_squared_error(a,data_test))
# r2=r2_score(a,data_test)

# sm.tsa.arma_order_select_ic(diff1,max_ar=6,max_ma=4,ic='aic')['aic_min_order']
# pmax = int(len(diff1) / 10)    #一般阶数不超过 length /10
# qmax = int(len(diff1) / 10)
# bic_matrix = []
# for p in range(pmax +1):
#     temp= []
#     for q in range(qmax+1):
#         try:
#             temp.append(ARIMA(diff1, (p, 1, q)).fit().bic)
#         except:
#             temp.append(None)
#         bic_matrix.append(temp)
#
# bic_matrix = pd.DataFrame(bic_matrix)   #将其转换成Dataframe 数据结构
# p,q = bic_matrix.stack().idxmin()   #先使用stack 展平， 然后使用 idxmin 找出最小值的位置
# print(u'BIC 最小的p值 和 q 值：%s,%s' %(p,q))  #  BIC 最小的p值 和 q 值：0,1