import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

data_final=pd.read_csv("../data/data_final.csv", index_col=0)
data_final.BTP.plot()