import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from minepy import MINE
import matplotlib.pyplot as plt

coef=[]
mine = MINE(alpha=0.6, c=15)
all=[]
for i in range(8,28,2):
    data_temp=pd.read_csv("../data_original/BTPandBRP2021-12-{0:02}-{1:02}.csv".format(i,i+2),index_col=0)
    data_process=pd.read_csv("../data_original/成分和工艺参数2021-12-{0:02}-{1:02}.csv".format(i,i+2),index_col=0)
    data_all=pd.concat([data_process,data_temp],axis=1)
    all.append(data_all)
data_alltogether=pd.concat(all,axis=0,ignore_index=True)
mid=list(data_process.columns)
mid.remove("time2")
variable_select=mid+list(data_temp.columns[-1:])
for n, i in enumerate(variable_select):
    x =data_alltogether[i].values;
    y =data_alltogether['BTP'].values;
    sx=np.concatenate([x,])
    pear_coef = pearsonr(x, y)[0]
    coef.append(pear_coef)

plt.figure()
plt.bar(range(len(coef)), np.abs(coef))

all=[]
for i in range(8,28,2):
    data_temp=pd.read_csv("../data_original/BTPandBRP2021-12-{0:02}-{1:02}.csv".format(i,i+2),index_col=0)
    data_process=pd.read_csv("../data_original/成分和工艺参数2021-12-{0:02}-{1:02}.csv".format(i,i+2),index_col=0)
    data_all=pd.concat([data_process,data_temp],axis=1)
    all.append(data_all)
data_alltogether=pd.concat(all,axis=0,ignore_index=True)
mine = MINE(alpha=0.6, c=15)
coef_mi = []
for n, i in enumerate(variable_select):
    x =data_alltogether[i].values;
    y =data_alltogether['BTP'].values;
    mine.compute_score(x, y)
    mi_coef = mine.mic()
    coef_mi.append(mi_coef)

plt.figure()
plt.bar(range(len(coef_mi)), np.abs(coef_mi))