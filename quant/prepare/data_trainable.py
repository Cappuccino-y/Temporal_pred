import akshare as ak
import pickle
from PyEMD import CEEMDAN,Visualisation
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
matplotlib.use('TkAgg')

with open("../data_roll_yield_f.pkl", "rb") as file:
    data_roll_yield= pickle.load(file)
with open("../data_spot_price_f.pkl", "rb") as file:
    data_spot_price = pickle.load(file)
with open("../data_rank_table_f.pkl", "rb") as file:
    data_rank_table= pickle.load(file)

non_zero_counts = (data_rank_table != 0).sum(axis=0)
top_institutions = non_zero_counts.nlargest(50).index
reduced_data_rank_table = data_rank_table[top_institutions]



data=data_spot_price.copy()
data['label']=data.dominant_contract_price.diff(1)/data.dominant_contract_price *100
data=data.drop(columns=['index', 'symbol','near_contract','date','dominant_contract'])
data=pd.concat([reduced_data_rank_table,data_roll_yield.roll_yield,data],axis=1)
data=data[1:].reset_index()

plot_acf(data.label, lags=50)  # lags参数指定滞后数
plt.title("ACF Plot")

# 绘制偏自相关函数（PACF）图
plot_pacf(data.label, lags=50) # lags参数指定滞后数
plt.title("PACF Plot")

ceemdan = CEEMDAN(trials=100)
ceemdan.ceemdan(data.dominant_contract_price.values)
imfs, res = ceemdan.get_imfs_and_residue()
vis = Visualisation()
vis.plot_imfs(imfs, res)



extend=[]
for i in range(imfs.shape[0]):
    extend.append(imfs[i].var()/data.dominant_contract_price.var())
extend=np.array(extend)

selec_num=8
index_max=np.argsort(-extend)
additional=pd.DataFrame(imfs[index_max[:selec_num]].transpose(1, 0), columns=["imfs_{}".format(col) for col in index_max[:selec_num]])
additional.to_csv("../data_train/data_append.csv")

data.to_csv("../data_train/data.csv")

