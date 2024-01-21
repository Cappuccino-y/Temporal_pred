import numpy as np
import pandas as pd
import pywt
import math
from math import log
from scipy.stats import pearsonr
# from statsmodels.tsa.stattools import acf, pacf
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from minepy import MINE
from PyEMD import CEEMDAN,Visualisation
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.stattools import ccf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# from dtaidistance import dtw
from sklearn.ensemble import IsolationForest
import matplotlib

def sgn(num):
    if (num>0.0):
        return 1.0
    elif (num==0.0):
        return 0.0
    else:
        return -1.0
all=[]
for i in range(8,30,2):
    index=[]
    data_temp=pd.read_csv("../data_original/BTPandBRP2021-12-{0:02}-{1:02}.csv".format(i,i+2),index_col=0)
    data_process=pd.read_csv("../data_original/成分和工艺参数2021-12-{0:02}-{1:02}.csv".format(i,i+2),index_col=0)
    data_all=pd.concat([data_process,data_temp],axis=1)
    all.append(data_all)

save=pd.read_csv("../data_original/成分和工艺参数2021-12-06-08.csv",index_col=0)
save_2=pd.read_csv("../data_original/BTPandBRP2021-12-06-08.csv",index_col=0)
save=pd.concat([save,save_2.simulation_BTP],axis=1)


data_alltogether=pd.concat(all,axis=0,ignore_index=True)
data_final=pd.concat([data_alltogether.iloc[:,:5],data_alltogether.iloc[:,6:9],data_alltogether.iloc[:,10],data_alltogether.iloc[:,-4]],axis=1)
data_revise=pd.read_csv("../data/data_revise_add.csv",index_col=0)
num=data_final.shape[0]
data_final=pd.concat([data_final.iloc[:,:-1],data_revise,data_final.iloc[:,-1]],axis=1)
data_final.drop(labels=range(31001,num),inplace=True)
num=data_final.shape[0]
x=0
def abnormal(value):
    return value==0

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
x=0
while(x<len(save)):
    if abnormal(save.simulation_BTP.values[x]):
        start=save.simulation_BTP.values[x-1]
        count=1
        while(abnormal(save.simulation_BTP.values[x+count])):
            count+=1
        end=save.simulation_BTP.values[x+count]
        for i in range(count):
            save.simulation_BTP.values[x+i]=(start+end)/2
            start=save.simulation_BTP.values[x+i]
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
save.rename(columns={'simulation_BTP': 'BTP'}, inplace=True)


# result=seasonal_decompose(data_final.BTP, model='additive',period=24*60)
# result.plot()
plt.figure()
plt.plot(data_final.iloc[:,-1])
# plt.figure()
# plt.plot(data_final.BRP)
# plt.ylim(60,120)
def wavelet_noising(new_df):
    data = new_df
    data = data.values.T.tolist()  # 将np.ndarray()转为列表
    w = pywt.Wavelet('dB10')#选择dB10小波基
    level=5
    coeffs = pywt.wavedec(data, w, level=level)  # 3层小波分解
    # ca3=ca3.squeeze(axis=0) #ndarray数组减维：(1，a)->(a,)
    # cd3 = cd3.squeeze(axis=0)
    # cd2 = cd2.squeeze(axis=0)
    # cd1 = cd1.squeeze(axis=0)
    # length1 = len(coeffs[-1])
    length0 = len(data)

    abs_cd1 = np.abs(np.array(coeffs[-1]))
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0 ), math.e))
    usecoeffs = []

    #软阈值方法
    coeffs.reverse()
    for index,seq in enumerate(coeffs[:-1]):
        for k in range(len(seq)):
            if (abs(seq[k]) >= lamda / np.log2(index+2)):
                seq[k] = sgn(seq[k]) * (abs(seq[k]) - lamda / np.log2(2))
            else:
                seq[k] = 0.0
        usecoeffs.append(seq)
    usecoeffs.append(coeffs[-1])
    usecoeffs.reverse()

    # for k in range(length1):
    #     if (abs(cd1[k]) >= lamda/np.log2(2)):
    #         cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - lamda/np.log2(2))
    #     else:
    #         cd1[k] = 0.0
    #
    # length2 = len(cd2)
    # for k in range(length2):
    #     if (abs(cd2[k]) >= lamda/np.log2(3)):
    #         cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - lamda/np.log2(3))
    #     else:
    #         cd2[k] = 0.0
    #
    # length3 = len(cd3)
    # for k in range(length3):
    #     if (abs(cd3[k]) >= lamda/np.log2(4)):
    #         cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - lamda/np.log2(4))
    #     else:
    #         cd3[k] = 0.0


    # usecoeffs.append(cd3)
    # usecoeffs.append(cd2)
    # usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)#信号重构
    if (len(recoeffs)!=len(new_df)):
        recoeffs=recoeffs[1:]
    return recoeffs

def anomaly_tackle(data,index):
    if (0 in index):
        index=index.drop(0)
    if (len(data)-1 in index):
        index = index.drop(len(data)-1)
    for i in index:
        start=i
        end=i
        while(start in index):
            start=start-1
        while(end in index):
            end=end+1
        data[i]=(data[start]+data[end])/2



plt.figure()
anomaly_ratio=[0.1]*12
anomaly_thre=[0]*12
for index,i in enumerate(data_final.columns.drop('BTP')):
    a=plt.subplot(6,4,2*index+1,title=i)
    plt.plot(data_final.loc[:,i])
    model = IsolationForest(n_estimators=50, max_samples='auto', contamination=float(anomaly_ratio[index]), max_features=1.0)
    model.fit(data_final[[i]])
    anomaly_score = model.decision_function(data_final[[i]])
    anomaly_index=data_final.index[anomaly_score<anomaly_thre[index]]
    anomaly_tackle(data_final.loc[:,i],anomaly_index)
    data_final.loc[:,i]=wavelet_noising(data_final.loc[:,i])
    b=plt.subplot(6,4,2*index+2,title=i+"_filterd")
    plt.plot(data_final.loc[:,i],color='r')
    b.set_ylim(a.get_ylim())



plt.show()

seed=64
mean=89.5
deviation=0.1
np.random.seed(seed)
# data_final.drop(data_final[ (data_final.BTP==78)|(data_final.BTP==90) ].index,inplace=True)
# data_final.drop(data_final[ (data_final.BRP<60)|(data_final.BTP>90) ].index,inplace=True)
for outlier_indice in data_final[(data_final.BTP==89.5)].index:
    data_final.BTP[outlier_indice]=np.random.normal(mean,deviation)
# data_final.drop(data_final[(data_final.BTP>89) | (data_final.BTP<84 )].index,inplace=True)

for outlier_indice in save[(save.BTP==89.5)].index:
    save.BTP[outlier_indice]=np.random.normal(mean,deviation)

# acf_50 = acf(data_final.BTP.values, nlags=50)
# pacf_50 = pacf(data_final.BTP.values, nlags=50)
# plt.figure()
# plt.plot(acf_50)
#
#
#
# mine = MINE(alpha=0.6, c=15)
# def cosSim(x,y):
#     '''
#     余弦相似度
#     '''
#     tmp=np.sum(x*y)
#     non=np.linalg.norm(x)*np.linalg.norm(y)
#     return np.round(tmp/float(non),9)
# def dtw_td(a,b,td,len_total,start_num,window_length):
#     res=[]
#     for i in range(td[0],td[1]+1):
#         # res.append(dtw.distance_fast(a[start_num+td-i:start_num+len_total+td-i],b[start_num+td:start_num+td+len_total]))
#         mine.compute_score(a[start_num+window_length-i:start_num+len_total+window_length-i],b[start_num+window_length:start_num+window_length+len_total])
#         res.append(mine.mic())
#         # res.append(cosSim(a[start_num+td-i:start_num+len_total+td-i],b[start_num+td:start_num+td+len_total]))
#         # k1=(a[start_num + td - i:start_num + len_total + td - i][1:]-a[start_num + td - i:start_num + len_total + td - i][:-1])
#         # k2=(b[start_num+td:start_num+td+len_total][1:]-b[start_num+td:start_num+td+len_total][:-1])
#         # res.append(2/(1+np.exp(np.sum(k1*k2)/np.linalg.norm(k1)/np.linalg.norm(k2)))*cosSim(a[start_num+td-i:start_num+len_total+td-i],b[start_num+td:start_num+td+len_total]))
#     return np.array(res).argmax()+td[0]
#
# data_reconstruct=data_final.copy()
#
# def re_td(a, b, delay):
#     if (delay==0):
#         return
#     prev= a[-delay:].values
#     now= b[:-delay].values
#     reconstruct= np.concatenate([prev, now])
#     b.iloc[:]=reconstruct
#
# delay_col=list(data_final.columns[0:7])+['ignition_temp']
# base_sup=45
# base_inf=35
# time_limit=5*[(base_inf+12,base_sup+12)]+[(base_inf+6,base_sup+6)]+[(base_inf,base_sup)]+[(base_inf,base_sup)]
# info=zip(delay_col,time_limit)
# delay_select=[]
# plt.figure()
# res_deley=[]
# save=save[delay_col+['BTP']]
# for index,(i,time_range) in enumerate(info):
#     mean_res = []
#     window_length=60
#     beta=0.8
#     length=480
#     prev_var=save.loc[:,i]
#     var=np.concatenate([save.loc[:,i].values[-(length):],data_final.loc[:,i]],axis=0)
#     tar=np.concatenate([save.BTP.values[-(length):],data_final.BTP.values],axis=0)
#     start_total=np.arange(0,len(var)-length-window_length,window_length)
#
#     # prev_a = dtw_td(np.concatenate([save.loc[:,i].values[-(length+window_length):],data_final.loc[:,i]],axis=0),
#     #                 np.concatenate([save.BTP.values[-(length+window_length):],data_final.BTP.values],axis=0), time_range, length, 0)
#     # prev_delay=prev_a
#     for start_num in start_total:
#         a=dtw_td(var,tar,time_range,length,start_num,window_length)
#         delay=a
#         re_td(prev_var,data_reconstruct.loc[start_num:start_num+window_length-1,i],delay)
#         prev_delay=delay
#         prev_var=data_final.loc[start_num:start_num+window_length-1,i]
#         mean_res.append(delay)
#     res_deley.append(np.array(mean_res))
    # plt.subplot(4,4,index+1,title=i)
    # plt.plot(mean_res)


# for index,i in enumerate(delay_col):
#     if (delay_select[index]==0):
#         continue
#     col_num=list(data_reconstruct.columns).index(i)
#     data_temp=save.iloc[-delay_select[index]:,col_num].values
#     origin=data_reconstruct.iloc[:-delay_select[index],col_num].values
#     col=np.concatenate([data_temp,origin])
#     data_reconstruct.iloc[:,col_num]=col



# result2=seasonal_decompose(data_final.BTP, model='additive',period=60)
# result2.plot()
#
matplotlib.use('TkAgg')
ceemdan = CEEMDAN(trials=100)
ceemdan.ceemdan(data_final.BTP.values)
imfs, res = ceemdan.get_imfs_and_residue()
vis = Visualisation()
vis.plot_imfs(imfs, res)
extend=[]
for i in range(imfs.shape[0]):
    extend.append(imfs[i].var()/data_final.BTP.var())
extend=np.array(extend)

selec_num=8
index_max=np.argsort(-extend)
additional=pd.DataFrame(imfs[index_max[:selec_num]].transpose(1, 0), columns=["imfs_{}".format(col) for col in index_max[:selec_num]])
# additional.to_csv("../data/data_append.csv")


# emd=EMD()
# emd(data_final.BTP.values)
# imfs, res = emd.get_imfs_and_residue()
# vis = Visualisation()
# t=np.arange(0,num)
# vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
# vis.plot_instant_freq(t, imfs=imfs)
# vis.show()



