import akshare as ak
import pickle
import pandas as pd
import time
start="20130101"
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
    # split_start_str = split_start.strftime("%Y%m%d")
    # split_end_str = split_end.strftime("%Y%m%d")
    current_date = split_start
    while current_date <= split_end:
        date_str = current_date.strftime("%Y%m%d")
        current_date += pd.Timedelta(days=1)  # 增加一天
        rank_table_= ak.get_shfe_rank_table(date=date_str,vars_list=[target])
        if rank_table_!={}:
            rank_table_['date']=date_str
            data.append(rank_table_)
    # if i!=len(date_range) - 2:
    #     time.sleep(30)

# rank_table=pd.concat(data,ignore_index=True)
with open("../data_rank_table.pkl", "wb") as file:
    pickle.dump(data,file)




# futures_shfe_warehouse_receipt_df = ak.futures_shfe_warehouse_receipt(trade_date="20120702")
# print(futures_shfe_warehouse_receipt_df)