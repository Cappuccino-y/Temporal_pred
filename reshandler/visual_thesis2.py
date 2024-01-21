import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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


def mape(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate absolute percentage error
    abs_percentage_error = np.abs((y_true - y_pred) / y_true)

    # Calculate mean of absolute percentage error
    mape_score = np.mean(abs_percentage_error,axis=0) * 100

    return mape_score


window_length=60
pred_length=7
data = pd.read_csv("../data_optimal/data_time_reconstruct.csv", index_col=0)
data_append=pd.read_csv("../data_optimal/data_append.csv",index_col=0)
feature_num_append=data_append.shape[1]

data=pd.concat([data_append,data],axis=1)
feature_num=len(data.columns)
target_num=1
sa_hidden=70
ta_hidden=70
temp_node=70
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


target = torch.from_numpy(data[:,data.shape[1]-target_num:]).float()
point=round(len(input)*0.9)
input_train_dev=input[:point]
target_train_dev=target[:point]
input_test=input[point:]
target_test=target[point:]

_,y_true_autoformer=windows_input(input_test,target_test,10,pred_length)
y_true_autoformer = y_true_autoformer.squeeze().numpy()
y_true_autoformer=y_true_autoformer[window_length-10:]

_, y_true = windows_input(input_test, target_test, window_length, pred_length)
y_true=y_true.squeeze().numpy()

# data_true=target_test[window_length+pred_length-1:]
data_true=pd.read_csv("../res_ARIMA/preds_true.csv", index_col=0)
# Arima
data_arima_pred=pd.read_csv("../res_ARIMA/preds_ARIMA_pred.csv", index_col=0)
rmse_arima=np.sqrt(mean_squared_error(data_true,data_arima_pred,multioutput='raw_values'))
r2_arima=r2_score(data_true,data_arima_pred,multioutput='raw_values')
mape_arima=mape(data_true,data_arima_pred)

# lstm
data_lstm_true=pd.read_csv("../res_lstm/pred_true_model_compare.csv", index_col=0)
data_lstm_pred=pd.read_csv("../res_lstm/preds_model_compare.csv", index_col=0)
rmse_lstm=np.sqrt(mean_squared_error(y_true,data_lstm_pred,multioutput='raw_values'))
r2_lstm=r2_score(y_true,data_lstm_pred,multioutput='raw_values')
mape_lstm=mape(y_true,data_lstm_pred)


# chapter1
data1_true=pd.read_csv("../res_Chapter1/pred_true_Chapter1.csv", index_col=0)
data1_pred=pd.read_csv("../res_Chapter1/preds_Chapter1.csv", index_col=0)
rmse_chapter1=np.sqrt(mean_squared_error(y_true,data1_pred,multioutput='raw_values'))
r2_chapter1=r2_score(y_true,data1_pred,multioutput='raw_values')
mape_chapter1=mape(y_true,data1_pred)

# chapter2
data2_true=pd.read_csv("../res_Chapter2/pred_true_Chapter2.csv", index_col=0)
data2_pred=pd.read_csv("../res_Chapter2/preds_Chapter2.csv", index_col=0)
rmse_chapter2=np.sqrt(mean_squared_error(y_true,data2_pred,multioutput='raw_values'))
r2_chapter2=r2_score(y_true,data2_pred,multioutput='raw_values')
mape_chapter2=mape(y_true,data2_pred)

# chapter3
data3_true=pd.read_csv("../res_Chapter3/pred_true_Chapter3.csv", index_col=0)
data3_pred=pd.read_csv("../res_Chapter3/preds_Chapter3.csv", index_col=0)
rmse_chapter3=np.sqrt(mean_squared_error(y_true,data3_pred,multioutput='raw_values'))
r2_chapter3=r2_score(y_true,data3_pred,multioutput='raw_values')
mape_chapter3=mape(y_true,data3_pred)

#chapter3 optimal
data3_optimal_true=pd.read_csv("../res_sampling/pred_true_Chapter3_hybrid.csv", index_col=0)
data3_optimal_pred=pd.read_csv("../res_sampling/preds_Chapter3_hybrid.csv", index_col=0)
rmse_chapter3_opt=np.sqrt(mean_squared_error(y_true,data3_optimal_pred,multioutput='raw_values'))
r2_chapter3_opt=r2_score(y_true,data3_optimal_pred,multioutput='raw_values')
mape_chapter3_opt=mape(y_true,data3_optimal_pred)

#LSTNet
data_LSTNet_true=pd.read_csv("../res_main/pred_true_main.csv",index_col=0)
data_LSTNet_pred=pd.read_csv("../res_main/preds_main.csv",index_col=0)
rmse_lstnet=np.sqrt(mean_squared_error(y_true,data_LSTNet_pred,multioutput='raw_values'))
r2_lstnet=r2_score(y_true,data_LSTNet_pred,multioutput='raw_values')
mape_lstnet=mape(y_true,data_LSTNet_pred)

#Autoformer
data_Autoformer_true=pd.read_csv("../res_Autoformer_predict/pred_true_Autoformer_predict.csv",index_col=0)
data_Autoformer_pred=pd.read_csv("../res_Autoformer_predict/preds_Autoformer_predict.csv",index_col=0)
rmse_autoformer=np.sqrt(mean_squared_error(y_true_autoformer,data_Autoformer_pred,multioutput='raw_values'))
r2_autoformer=r2_score(y_true_autoformer,data_Autoformer_pred,multioutput='raw_values')
mape_autoformer=mape(y_true_autoformer,data_Autoformer_pred)

plot_num=-1
# num=data_arima_pred.shape[0]
num=700
deviation= 160 - 60
deviation_autoformer=160-10
# Font style dictionary for title
title_font = {'fontname':'Times New Roman', 'size':'8', 'color':'black', 'weight':'normal'}
# Font style dictionary for labels
label_font = {'fontname':'Times New Roman', 'size':'6', 'color':'black', 'weight':'normal'}
label_font_y={'fontname':'Times New Roman', 'size':'12', 'color':'black', 'weight':'normal'}
# Font style for legend
legend_font = fm.FontProperties(family='SimSun', size=10)
Plotdata=[data_arima_pred, data_lstm_pred, data1_pred, data_LSTNet_pred, data_Autoformer_pred, data2_pred,data3_pred]
Plotname=['ARIMA','LSTM','TA-LSTM','LSTNet','Autoformer','TSE-TSD','TSE-TSTD']
Plotcomb=zip(Plotname, Plotdata)
figure_num=len(Plotdata)
fig, axs = plt.subplots(figure_num, 1, figsize=(10,20))
for index,(name,pred) in enumerate(Plotcomb):
    axs[index].plot(data_true.iloc[:num, -1][:plot_num],color='r')
    if name.find('ARIMA')>-1:
        axs[index].plot(pred.iloc[:num,-1][:plot_num])
    # elif name.find('Autoformer')>-1:
    #     axs[index].plot(pred.iloc[deviation_autoformer:deviation_autoformer+num, -1][:plot_num])
    else:
        axs[index].plot(pred.iloc[deviation:deviation + num, -1].values[:plot_num])
    axs[index].legend(['真实值','预测值'],  prop=legend_font,frameon=False)
    axs[index].set_xlim(-20, num+20)
    axs[index].set_ylim(80,100)
    axs[index].set_ylabel('BTP',**label_font_y)
    # axs[index].set_xlabel('Samples',**label_font)
    axs[index].text(300-1.8*len('{}测试集第7步预测结果'.format(name)),97 , '{}测试集第7步预测结果'.format(name), fontsize=12,
                fontname='SimSun')
    # axs[index].set_title('Prediction results with {}'.format(name),**title_font)

# plt.tight_layout()
# plt.subplots_adjust(hspace=2.0)
plt.tight_layout()
plt.show(block=True)

# deviation= 160 - 60
# num=data_arima_pred.shape[0]
# plot_num=-1
# plt.figure()
# plt.plot(data_true.iloc[:,-1][:plot_num])
# plt.plot(data_arima_pred.iloc[:,-1][:plot_num])
# plt.plot(data_lstm_pred.iloc[deviation:deviation + num, -1].values[:plot_num])
# plt.plot(data1_pred.iloc[deviation:deviation + num, -1].values[:plot_num])
#
# plt.legend(['BTP','ARIMA','LSTM','TA-ARLSTM'])
# plt.ylim(82,90)
#
#
# plt.figure()
# plt.plot(data_true.iloc[:,-1][:plot_num])
# plt.plot(data2_pred.iloc[deviation:deviation + num, -1].values[:plot_num])
# plt.plot(data3_pred.iloc[deviation:deviation + num, -1].values[:plot_num])
# plt.plot(data3_optimal_pred.iloc[deviation:deviation + num, -1].values[:plot_num])
# plt.legend(['BTP','CCF-TCN Enhanced TA-ARLSTM','CEEMD CCF-TCN Enhanced TA-ARLSTM','Multi-tasking CEEMD CCF-TCN Enhanced TA-ARLSTM '])
# plt.ylim(82,90)
# data_true=pd.read_csv("../res/pred_true.csv",index_col=0)
# data_pred=pd.read_csv("../res/preds.csv",index_col=0)
# plt.plot(data_pred['0'])
# plt.plot(data_true['0'])