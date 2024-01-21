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
start_date = pd.to_datetime(start, format="%Y%m%d")
end_date = pd.to_datetime(end, format="%Y%m%d")


with open("../data_rank_table.pkl", "rb") as file:
    data_update=pickle.load(file)

current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime("%Y%m%d")
    current_date += pd.Timedelta(days=1)  # 增加一天
    rank_table_= ak.get_shfe_rank_table(date=date_str,vars_list=[target])
    if rank_table_!={}:
        rank_table_['date']=date_str
        data_update.append(rank_table_)

with open("../data_rank_table.pkl", "wb") as file:
    pickle.dump(data_update,file)




# futures_shfe_warehouse_receipt_df = ak.futures_shfe_warehouse_receipt(trade_date="20120702")
# print(futures_shfe_warehouse_receipt_df)