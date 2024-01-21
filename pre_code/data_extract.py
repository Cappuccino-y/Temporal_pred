import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from minepy import MINE
from PyEMD import CEEMDAN,Visualisation
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import ccf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from dtaidistance import dtw

all=[]
for i in range(8,30,2):
    index=[]
    data_temp=pd.read_csv("../data_original/BTPandBRP2021-12-{0:02}-{1:02}.csv".format(i,i+2),index_col=0)
    data_process=pd.read_csv("../data_original/成分和工艺参数2021-12-{0:02}-{1:02}.csv".format(i,i+2),index_col=0)
    data_all=pd.concat([data_process,data_temp],axis=1)
    all.append(data_all)

save=pd.read_csv("../data_original/成分和工艺参数2021-12-06-08.csv",index_col=0)
save=pd.concat([save.iloc[:,:5],save.iloc[:,6:9],save.iloc[:,10]],axis=1)


data_alltogether=pd.concat(all,axis=0,ignore_index=True)
data_final=pd.concat([data_alltogether.iloc[:,:5],data_alltogether.iloc[:,6:9],data_alltogether.iloc[:,10],data_alltogether.iloc[:,-4]],axis=1)
data_revise=pd.read_csv("../data/data_revise_add.csv",index_col=0)
num=data_final.shape[0]
data_final=pd.concat([data_final.iloc[:,:-1],data_revise,data_final.iloc[:,-1]],axis=1)
data_final.drop(labels=range(31001,num),inplace=True)
num=data_final.shape[0]
x=0
def abnormal(value):
    return value<82

while(x<num):
    if abnormal(data_final.simulation_BTP.values[x]):
        start=data_final.simulation_BTP.values[x-1]
        count=1
        while(abnormal(data_final.simulation_BTP.values[x+count])):
            count+=1
        end=data_final.simulation_BTP.values[x+count]
        for i in range(count):
            data_final.simulation_BTP.values[x+i]=(start+end)/2
            start=data_final.simulation_BTP.values[x+i]
        x=x+count
    else:
        x+=1

# while(x<num):
#     if data_final.simulation_BTP.values[x]<82:
#         start=data_final.simulation_BTP.values[x-1]
#         count=1
#         while(data_final.simulation_BTP.values[x+count]<82):
#             count+=1
#         end=data_final.simulation_BTP.values[x+count]
#         for i in range(count):
#             data_final.simulation_BTP.values[x+i]=(start+end)/2
#             start=data_final.simulation_BTP.values[x+i]
#         x=x+count
#     else:
#         x+=1

data_final.rename(columns={'simulation_BTP': 'BTP'}, inplace=True)


# result=seasonal_decompose(data_final.BTP, model='additive',period=24*60)
# result.plot()
plt.figure()
plt.plot(data_final.iloc[:,-1])
# plt.figure()
# plt.plot(data_final.BRP)
# plt.ylim(60,120)
seed=64
mean=89.5
deviation=0.1
np.random.seed(seed)
# data_final.drop(data_final[ (data_final.BTP==78)|(data_final.BTP==90) ].index,inplace=True)
# data_final.drop(data_final[ (data_final.BRP<60)|(data_final.BTP>90) ].index,inplace=True)
for outlier_indice in data_final[(data_final.BTP==89.5)].index:
    data_final.BTP[outlier_indice]=np.random.normal(mean,deviation)
# data_final.drop(data_final[(data_final.BTP>89) | (data_final.BTP<84 )].index,inplace=True)

acf_50 = acf(data_final.BTP.values, nlags=50)
pacf_50 = pacf(data_final.BTP.values, nlags=50)
plt.figure()
plt.plot(acf_50)

manhattan_distance = lambda x, y: np.abs(x - y)
delay_col=list(data_final.columns[0:7])+['ignition_temp']
time_limit=[60]+6*[60]+[50]
info=zip(delay_col,time_limit)
delay_select=[]
plt.figure()

mine = MINE(alpha=0.6, c=15)
def dtw_td(a,b,td,len_total,start_index):
    res=[]
    for i in range(td):
        # res.append(dtw.distance_fast(a[start_index+i:start_index+len_total+i],b[start_index:start_index+len_total]))
        mine.compute_score(a[start_index+i:start_index+len_total+i],b[start_index:start_index+len_total])
        res.append(mine.mic())
    return np.array(res)

# def re_td(delay,)

count=np.zeros(len(delay_col))
for index,(i,time_range) in enumerate(info):
    # a=ccf(data_final.loc[:,i].values,data_final.BTP.values)
    # judge=a[:time_range]
    # judge=-np.sign(judge[0])*judge
    # delay_select.append(judge.argsort()[0])
    # plt.subplot(4,4,index+1,title=i)
    # plt.xlim(0,time_range)
    # plt.ylim(a[0]-0.05,a[0]+0.05)
    #
    #
    # plt.plot(a)
    length=360
    fold_num=50
    sample_num=500
    start_total=[]
    for fold in range(fold_num):
        start = round(fold * (len(data_final)-time_range-length) / fold_num)
        end = round((fold + 1) * (len(data_final)-time_range-length) / fold_num) if (fold + 1) * (len(data_final)-time_range-length) / 10 <= (len(
            data_final)-time_range-length) else (len(data_final)-time_range-length)
        start_total=np.concatenate([start_total,np.random.choice(np.arange(start,end,5),round(sample_num/fold_num),replace=False)])
    start_total=start_total.astype('int')
    mean_res=np.zeros(time_range)
    # mean_res=[]
    for start_num in start_total:
        a=dtw_td(data_final.loc[:,i].values,data_final.BTP.values,time_range,length,start_num)
        judge=a
        if judge.argsort()[0]>=0:
            # mean_res.append(judge.argsort()[0])
            ss=StandardScaler()
            mean_res += ss.fit_transform(a.reshape(-1,1)).reshape(-1)
        else:
            count[index]+=1
    # delay_select.append(round(np.mean(mean_res)))
    delay_select.append(mean_res.argsort()[0])


data_reconstruct=data_final.copy()
for index,i in enumerate(delay_col):
    if (delay_select[index]==0):
        continue
    col_num=list(data_reconstruct.columns).index(i)
    data_temp=save.iloc[-delay_select[index]:,col_num].values
    origin=data_reconstruct.iloc[:-delay_select[index],col_num].values
    col=np.concatenate([data_temp,origin])
    data_reconstruct.iloc[:,col_num]=col



result2=seasonal_decompose(data_final.BTP, model='additive',period=60)
result2.plot()
#

# ceemdan = CEEMDAN(trials=100)
# ceemdan.ceemdan(data_final.BTP.values)
# imfs, res = ceemdan.get_imfs_and_residue()
# vis = Visualisation()
# vis.plot_imfs(imfs, res)
# extend=[]
# for i in range(imfs.shape[0]):
#     extend.append(imfs[i].var()/data_final.BTP.var())
# extend=np.array(extend)
#
# selec_num=8
# index_max=np.argsort(-extend)
# additional=pd.DataFrame(imfs[index_max[:selec_num]].transpose(1, 0), columns=["imfs_{}".format(col) for col in index_max[:selec_num]])
# additional.to_csv("../data/data_append.csv")

# data_reconstruct.to_csv("../data/data_time_reconstruct.csv")

# emd=EMD()
# emd(data_final.BTP.values)
# imfs, res = emd.get_imfs_and_residue()
# vis = Visualisation()
# t=np.arange(0,num)
# vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
# vis.plot_instant_freq(t, imfs=imfs)
# vis.show()

plt.figure()
plt.plot(data_final.values[:,-1])
data_final.to_csv("../data/data_final.csv")

