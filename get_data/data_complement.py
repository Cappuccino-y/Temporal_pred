from influxdb import InfluxDBClient
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy import signal

client = InfluxDBClient(host="172.16.12.103", port="32310",
                        username="admin", password="123456",
                        database="sjtu")
# T_start='\'2021-11-17T02:17:00Z\''
# T_end='\'2021-11-18T02:17:00Z\''

T_start = '\'2021-12-08T02:17:00Z\''
T_end = '\'2021-12-30T02:16:00Z\''

sql=[]
res=[]
# 大烟道南侧温度 负压 南侧24号风箱温度 点火温度DATA005
sql.append( '''SELECT DATA128,DATA134,DATA176 FROM "IPLATURE.BAVERAGE_13AVIEW" WHERE time >= {} and   time <= {} ;'''.format(T_start,T_end))

#南侧24号风箱负压
# sql.append( '''SELECT DATA056 FROM "IPLATURE.BAVERAGE_14AVIEW" WHERE time >= {} and   time <= {} ;'''.format(T_start,T_end))


for item in sql:
    temp= pd.DataFrame(client.query(item).get_points())
    res.append(temp.iloc[:,1:])

res=np.concatenate(res,axis=1)
print(res.shape[0])

res=pd.DataFrame(res,columns=["main_temp","main_pressure","No.24_temp"])
res.to_csv("../data/data_revise_add.csv")



