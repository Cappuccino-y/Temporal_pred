import akshare as ak
import pickle
import pandas as pd
import datetime

with open('./lastpoint.pkl','rb') as file:
    lastpoint=pickle.load(file)

start=(pd.to_datetime(lastpoint)+pd.Timedelta('1 day')).strftime('%Y%m%d')
end = datetime.datetime.now().date().strftime('%Y%m%d')

target="AU"
# 将字符串转换为日期对象
# start_date = pd.to_datetime(start, format="%Y%m%d")
# end_date = pd.to_datetime(end, format="%Y%m%d")


with open("../data_spot_price.pkl", "rb") as file:
    data_update=pickle.load(file)

spot_price_ = ak.futures_spot_price_daily(start_day=start, end_day=end, vars_list=[target])
if not spot_price_.empty:
    data_update.append(spot_price_)

with open("../data_spot_price.pkl", "wb") as file:
    pickle.dump(data_update,file)