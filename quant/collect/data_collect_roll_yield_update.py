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


with open("../data_roll_yield.pkl", "rb") as file:
    data_update=pickle.load(file)

roll_yield_=ak.get_roll_yield_bar(type_method="date", var=target, start_day=start, end_day=end)
if not roll_yield_.empty:
    data_update.append(roll_yield_)

with open("../data_roll_yield.pkl", "wb") as file:
    pickle.dump(data_update,file)




# futures_shfe_warehouse_receipt_df = ak.futures_shfe_warehouse_receipt(trade_date="20120702")
# print(futures_shfe_warehouse_receipt_df)