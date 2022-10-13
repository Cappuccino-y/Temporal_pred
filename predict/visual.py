import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from minepy import MINE
import matplotlib.pyplot as plt

data_final=pd.read_csv("../data/data_final.csv",index_col=0)

data_final=data_final.iloc[:,:-1]
c=[]
for i in data_final.columns:
    c.append(i)
data_final.plot(subplots=True, color="r", layout=(3, 4),
                     title=c)
