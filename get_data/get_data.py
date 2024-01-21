from influxdb import InfluxDBClient
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy import signal

import sys


def rough_show(data, i):
    plt.plot(data[i])


def fit_curve(x, A, B, C):
    return A + B * x + C * x ** 2


def targe_fuc_max(para):
    def inner(x):
        target = -(para[0] + para[1] * x[0] + para[2] * x[0] ** 2)
        return target

    return inner


# 初始化
client = InfluxDBClient(host="172.16.12.103", port="32310",
                        username="admin", password="123456",
                        database="sjtu")
start_num = 20  # must >= 5
# T_start='\'2021-11-17T02:17:00Z\''
# T_end='\'2021-11-18T02:17:00Z\''
T_start = '\'2021-12-12T22:17:00Z\''
T_end = '\'2021-12-16T06:17:00Z\''
# sql1 = '''select * from "IPLATURE.BAVERAGE_13AVIEW" order by time desc limit 100'''
# time >= '2021-11-19T02:17:00Z'and time <= '2021-11-29T02:17:00Z' ;'''
# sql1 ='''SELECT D FROM "IPLATURE.BAVERAGE_13AVIEW" WHERE time >= '2021-11-28T02:17:00Z' ;'''
# sql1 ='''SELECT DATA033 FROM "IPLATURE.BAVERAGE_13AVIEW" WHERE time >= '2021-11-19T02:17:00Z' and   time <= '2021-11-29T02:17:00Z' ;'''
sql1 = '''SELECT DATA125,DATA127 FROM "IPLATURE.BAVERAGE_13AVIEW" WHERE time >= {} and   time <= {} ;'''.format(T_start,
                                                                                                                T_end)
sql2 = '''SELECT DATA057 FROM "IPLATURE.BAVERAGE_14AVIEW" WHERE time >= {} and   time <= {} ;'''.format(T_start, T_end)
# sql3='''SELECT DATA057 FROM "IPLATURE.BAVERAGE_14AVIEW" WHERE time >= {} and   time <= {} ;'''.format(T_start,T_end)
sql4 = '''SELECT DATA058,DATA059,DATA60,DATA61,DATA62,DATA63,DATA064,DATA065,DATA066
 FROM "IPLATURE.BAVERAGE_14AVIEW" WHERE time >= {} and   time <= {} ;'''.format(T_start, T_end)
sql5 = '''SELECT DATA179 FROM "IPLATURE.BAVERAGE_13AVIEW" WHERE time >= {} and   time <= {} ;'''.format(T_start, T_end)
sql6 = '''SELECT DATA067,DATA068,DATA069,DATA070,DATA071,DATA072,DATA073 FROM "IPLATURE.BAVERAGE_14AVIEW" WHERE time >= {} and   time <= {} ;'''.format(
    T_start, T_end)
sql7 = '''SELECT DATA180,DATA181,DATA182 FROM "IPLATURE.BAVERAGE_13AVIEW" WHERE time >= {} and   time <= {} ;'''.format(
    T_start, T_end)
res1 = pd.DataFrame(client.query(sql1).get_points())
res2 = pd.DataFrame(client.query(sql2).get_points())
res4 = pd.DataFrame(client.query(sql4).get_points())
res5 = pd.DataFrame(client.query(sql5).get_points())
res6 = pd.DataFrame(client.query(sql6).get_points())
res7 = pd.DataFrame(client.query(sql7).get_points())

res = pd.concat(
    [res1.iloc[:, 1:], res2.iloc[:, 1:], res4.iloc[:, 1:], res5.iloc[:, 1:], res6.iloc[:, 1:], res7.iloc[:, 1:]],
    axis=1)
print(res)
res = res.to_numpy().astype('float64')
data_x_1 = res[:, [0, 15, 19]]
res = res[:, start_num - 2:]

p0 = [5770, -7900, 360]
Para = []
for i in range(res.shape[0]):
    Para.append(curve_fit(fit_curve, np.arange(start_num, 25), res[i, :])[0])
btp = []
x0 = 23
limit = 24
lowest = 20
for p in Para:
    btp.append(minimize(targe_fuc_max(p), x0, bounds=[(0, limit)]).x)

# save_data=pd.DataFrame(res,np.arange(0,res.shape[0]))
# save_data.to_csv("E:/mfile/sinter/data.csv",index=None,header=None)
plt.ylim([0, limit])
btp = np.array(btp)
for i in range(btp.shape[0]):
    btp[i, 0] = btp[i, 0] if lowest <= btp[i, 0] <= limit else btp[i - 1, 0]
data_y = btp * 90 / 24
plt.plot(btp)
plt.grid()
plt.ylim([lowest, limit])

sql_press_1th = '''SELECT DATA131 FROM "IPLATURE.BAVERAGE_13AVIEW" WHERE time >= {} and   time <= {} ;'''.format(
    T_start, T_end)
sql_press_17th = '''SELECT DATA086 FROM "IPLATURE.BAVERAGE_14AVIEW" WHERE time >= {} and   time <= {} ;'''.format(
    T_start, T_end)
sql_press_21th = '''SELECT DATA090 FROM "IPLATURE.BAVERAGE_14AVIEW" WHERE time >= {} and   time <= {} ;'''.format(
    T_start, T_end)
sql_temp_pipe = '''SELECT DATA129 FROM "IPLATURE.BAVERAGE_13AVIEW" WHERE time >= {} and   time <= {} ;'''.format(
    T_start, T_end)
sql_press_pipe = '''SELECT DATA135 FROM "IPLATURE.BAVERAGE_13AVIEW" WHERE time >= {} and   time <= {} ;'''.format(
    T_start, T_end)
sql_speed_move = '''SELECT DATA005 FROM "IPLATURE.BAVERAGE_13AVIEW" WHERE time >= {} and   time <= {} ;'''.format(
    T_start, T_end)
sql_speed_cylinder = '''SELECT DATA196 FROM "IPLATURE.BAVERAGE_13AVIEW" WHERE time >= {} and   time <= {} ;'''.format(
    T_start, T_end)
sql_thick = '''SELECT DATA003 FROM "IPLATURE.BAVERAGE_13AVIEW" WHERE time >= {} and   time <= {} ;'''.format(T_start,
                                                                                                             T_end)
sql_volume = '''SELECT DATA186 FROM "IPLATURE.BAVERAGE_13AVIEW" WHERE time >= {} and   time <= {} ;'''.format(T_start,
                                                                                                              T_end)
sql_damper = '''SELECT DATA188 FROM "IPLATURE.BAVERAGE_13AVIEW" WHERE time >= {} and   time <= {} ;'''.format(T_start,
                                                                                                              T_end)
sql_temp_fire = '''SELECT DATA167 FROM "IPLATURE.BAVERAGE_13AVIEW" WHERE time >= {} and   time <= {} ;'''.format(
    T_start, T_end)
sql_water = '''SELECT DATA002 FROM "IPLATURE.BAVERAGE_13AVIEW" WHERE time >= {} and   time <= {} ;'''.format(T_start,
                                                                                                             T_end)

# data_press_1th = pd.DataFrame(client.query(sql_press_1th).get_points())
data_press_17th = pd.DataFrame(client.query(sql_press_17th).get_points())
data_press_21th = pd.DataFrame(client.query(sql_press_21th).get_points())
data_temp_pipe = pd.DataFrame(client.query(sql_temp_pipe).get_points())
data_press_pipe = pd.DataFrame(client.query(sql_press_pipe).get_points())
data_speed_move = pd.DataFrame(client.query(sql_speed_move).get_points())
data_speed_cylinder = pd.DataFrame(client.query(sql_speed_cylinder).get_points())
data_thick = pd.DataFrame(client.query(sql_thick).get_points())
data_volume = pd.DataFrame(client.query(sql_volume).get_points())
data_damper = pd.DataFrame(client.query(sql_damper).get_points())
data_temp_fire = pd.DataFrame(client.query(sql_temp_fire).get_points())
data_water = pd.DataFrame(client.query(sql_water).get_points())

# data_press_1th.iloc[:, 1:], data_press_17th.iloc[:, 1:],
data_x_2 = pd.concat([
    data_press_21th.iloc[:, 1:], data_temp_pipe.iloc[:, 1:],
    data_press_pipe.iloc[:, 1:], data_speed_move.iloc[:, 1:],
    data_speed_cylinder.iloc[:, 1:], data_thick.iloc[:, 1:],
    data_volume.iloc[:, 1:], data_damper.iloc[:, 1:],
    data_temp_fire.iloc[:, 1:], data_water.iloc[:, 1:]], axis=1)
# "DATA131": "1号风箱负压",
data_x_2 = data_x_2.rename(columns={"DATA086": "17号风箱负压", "DATA090": "21号风箱负压", "DATA129": "大烟道温度",
                                    "DATA135": "大烟道负压", "DATA005": "烧结机移速", "DATA196": "七辊速度", "DATA003": "料层厚度",
                                    "DATA186": "风机风量", "DATA188": "风门开度", "DATA167": "点火温度", "DATA002": "水分含量"})
data_x_1 = pd.DataFrame(data_x_1, columns=["1号风箱温度", "17号风箱温度", "21号风箱温度"])

data_x = pd.concat([data_x_1, data_x_2], axis=1)
data_y = pd.DataFrame(data_y, columns=["实际终点位置"])
data = pd.concat([data_x, data_y], axis=1)
data.to_csv("data.csv", encoding="utf_8_sig", index=None)

b, a = signal.butter(4, 0.1, 'lowpass')
filtedData = signal.filtfilt(b, a, data_x.values, axis=0)
x_filted = pd.DataFrame(filtedData, columns=data_x.columns)
data_filted = pd.concat([x_filted, data_y], axis=1)
data_filted.to_csv("data_filted.csv", encoding="utf_8_sig", index=None)
