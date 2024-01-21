import pandas as pd
import pickle

with open("../data_roll_yield.pkl", "rb") as file:
    data_roll_yield= pickle.load(file)
with open("../data_spot_price.pkl", "rb") as file:
    data_spot_price = pickle.load(file)
with open("../data_rank_table.pkl", "rb") as file:
    data_rank_table= pickle.load(file)

duplicates = data_spot_price['date'].duplicated(keep='first')
data_spot_price=data_spot_price[~duplicates].reset_index()



date_rank=[]
date_price=set(data_spot_price.date)

data_rank_table_removedup=[]
for i in range(len(data_rank_table)):
    date=data_rank_table[i]['date']
    if date not in date_rank and date in date_price:
        date_rank.append(date)
        data_rank_table_removedup.append(data_rank_table[i])

diff=list(date_price-set(date_rank)).sort()

data_roll_yield=data_roll_yield[-data_spot_price.shape[0]:].reset_index()

df_list= []

# 遍历原始列表
for i,d in enumerate(data_rank_table_removedup):
    if d:  # 确保字典不是空的

        value=d[data_spot_price.loc[i].dominant_contract]
        # 将键值对添加到新列表中
        df_list.append(value)

result_df = pd.DataFrame()

# 遍历列表中的每个DataFrame
for df in df_list:
    # 将vol_party_name的值转换为列，vol列的值作为新列的数据
    # 这里使用了groupby和first来确保每个vol_party_name只产生一列
    df['vol_party_name'] = df['vol_party_name'].fillna('others')
    transformed_df = df.groupby('vol_party_name')['vol'].first().to_frame()

    # 将转换后的单行DataFrame添加到结果DataFrame中
    result_df = pd.concat([result_df, transformed_df.transpose()], ignore_index=True)

result_df.fillna(0, inplace=True)

with open("../data_spot_price_f.pkl", "wb") as file:
    pickle.dump(data_spot_price,file)
with open("../data_roll_yield_f.pkl", "wb") as file:
    pickle.dump(data_roll_yield,file)
with open("../data_rank_table_f.pkl", "wb") as file:
    pickle.dump(result_df,file)