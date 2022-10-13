import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from minepy import MINE
import matplotlib.pyplot as plt

for i in range(6,32,2):
    data_temp=pd.read_csv("../data_original/BTPandBRP2021-12-{0:02}-{1:02}.csv".format(i,i+2),index_col=0)
    plt.figure()
    plt.plot(data_temp["BTP"])

coef=[]
mine = MINE(alpha=0.6, c=15)
for i in range(6,32,2):
    data_temp=pd.read_csv("../data_original/BTPandBRP2021-12-{0:02}-{1:02}.csv".format(i,i+2),index_col=0)
    variable_select=data_temp.columns[-1:]
    for n, i in enumerate(variable_select):
        x =data_temp[i].values;
        y =data_temp['BTP'].values;
        pear_coef = pearsonr(x, y)[0]
        coef.append(pear_coef)

plt.figure()
plt.bar(range(len(coef)), np.abs(coef))

mine = MINE(alpha=0.6, c=15)
coef_mi = []
variables_objects_mi = []
variables_nan_mi = []
for i in range(6,32,2):
    data_temp=pd.read_csv("../data_original/BTPandBRP2021-12-{0:02}-{1:02}.csv".format(i,i+2),index_col=0)
    variable_select=data_temp.columns[-1:]
    for n, i in enumerate(variable_select):
        x =data_temp[i].values;
        y =data_temp['BTP'].values;
        mine.compute_score(x, y)
        mi_coef = mine.mic()
        coef_mi.append(mi_coef)

plt.figure()
plt.bar(range(len(coef_mi)), np.abs(coef_mi))