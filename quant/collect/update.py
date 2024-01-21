import akshare as ak
import pickle
import pandas as pd
import time
import torch
import os
import datetime
import chinese_calendar as calendar
import pickle
import sys
import subprocess

with open('./lastpoint.pkl','rb') as file:
    lastone=pickle.load(file)

now= datetime.datetime.now().date().strftime('%Y%m%d')
if pd.to_datetime(lastone)>pd.to_datetime(now):
    print("error occurs")
elif pd.to_datetime(lastone)==pd.to_datetime(now):
    print("No need")
else:
    subprocess.run(['python', 'data_collect_rank_inventory_update.py'])
    subprocess.run(['python', 'data_collect_roll_yield_update.py'])
    subprocess.run(['python', 'data_collect_spot_price_update.py'])

    subprocess.run(['python', '../prepare/preprocess.py'], cwd='../prepare')
    subprocess.run(['python', '../prepare/data_trainable.py'], cwd='../prepare')
    with open('./lastpoint.pkl','wb') as file:
        pickle.dump(now, file)