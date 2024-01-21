import akshare as ak
import pickle
import pandas as pd
import time
start="20110104"
end="20240120"
target="AU"
# 将字符串转换为日期对象
start_date = pd.to_datetime(start, format="%Y%m%d")
end_date = pd.to_datetime(end, format="%Y%m%d")

# 定义时间间隔，例如 '1M' 代表一个月
interval = '6M'

# 生成日期范围
date_range = pd.date_range(start=start_date, end=end_date, freq=interval)
date_range=date_range.to_list()
date_range.append(end_date)

data=[]

for i in range(len(date_range) - 1):
    split_start = date_range[i]
    split_end = date_range[i + 1]
    split_start_str = split_start.strftime("%Y%m%d")
    split_end_str = split_end.strftime("%Y%m%d")
    roll_yield_=ak.get_roll_yield_bar(type_method="date", var=target, start_day=split_start_str, end_day=split_end_str)
    data.append(roll_yield_)
    if i!=len(date_range) - 2:
        time.sleep(60)

roll_yield=pd.concat(data,ignore_index=True)
with open("../data_roll_yield.pkl", "wb") as file:
    pickle.dump(roll_yield,file)