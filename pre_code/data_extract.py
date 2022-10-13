import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from minepy import MINE
import matplotlib.pyplot as plt

all=[]
for i in range(12,26,2):
    data_temp=pd.read_csv("../data_original/BTPandBRP2021-12-{0:02}-{1:02}.csv".format(i,i+2),index_col=0)
    data_process=pd.read_csv("../data_original/成分和工艺参数2021-12-{0:02}-{1:02}.csv".format(i,i+2),index_col=0)
    data_all=pd.concat([data_process,data_temp],axis=1)
    all.append(data_all)
data_alltogether=pd.concat(all,axis=0,ignore_index=True)
data_final=pd.concat([data_alltogether.iloc[:,:5],data_alltogether.iloc[:,6:12],data_alltogether.iloc[:,-1],data_alltogether.iloc[:,-2]],axis=1)
data_final.drop(data_final[(data_final.BTP>89) | (data_final.BTP<84 )].index,inplace=True)
data_final.to_csv("../data/data_final.csv")

