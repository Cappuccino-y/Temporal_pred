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
index_us_stock_sina_df = ak.index_us_stock_sina(symbol=".INX")
print(index_us_stock_sina_df)