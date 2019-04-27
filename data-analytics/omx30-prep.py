#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
import sys

import warnings
import math

df=pd.read_csv('../data/data.csv.gz', compression='gzip', sep=',')
df['timestamp'] = pd.to_datetime(df['time_stamp'], format="%Y-%m-%d %H:%M:%S").dt.tz_convert('CET')


# In[56]:


stock_delta = df['stock_id'] - df['stock_id'].shift()
time_delta = (df['timestamp'] - df['timestamp'].shift()).fillna(10000).dt.total_seconds()


# In[57]:


stock_split_index=df.index[stock_delta!=0].tolist()
stock_split_index.append(len(df))


# In[58]:


day_split_index=df.index[abs(time_delta)>36000].tolist()
day_split_index.insert(0, 0)
day_split_index.append(len(df))


# In[59]:


stock_id_list = df['stock_id'].unique().tolist()
i = 0
for stock_index in stock_id_list:
    
    print("index: {}, id:{}".format(i, stock_index))
    i+=1


# In[60]:


current_stock_index = 0
time_series_list = []
time_series_all_list = []
for i in range(len(day_split_index) - 1):
    start_index = day_split_index[i]
    end_index = day_split_index[i + 1]
    day_time_series = df.iloc[start_index:end_index]
    time_series_list.append(day_time_series)
    if end_index == stock_split_index[current_stock_index + 1]:
        print("finished stock: {}, got {} time series".format(current_stock_index, len(time_series_list)))
        time_series_all_list.append(time_series_list)
        time_series_list = []
        current_stock_index += 1


# In[61]:


for stock_id in range(len(stock_id_list)):
    diff = time_series_all_list[stock_id][0]['open'].diff()

    min_ = diff[diff>=0.005].min()
    open_ = time_series_all_list[stock_id][0]['open'].mean()

    print("stockid:{} min={}, cost={}".format(stock_id, min_, min_/open_))


# In[63]:


for stock_id in range(len(time_series_all_list)):
    print("handling stock_id: {}".format(stock_id))
    for day_id in range(len(time_series_all_list[stock_id])):
        # :-1 is that we don't like the last record at 17:30 which is a aggregated number.
        df = time_series_all_list[stock_id][day_id]
        # some data might miss, we must make a right join with full time series
        # and do fillna.
        df2 = df.set_index('timestamp')
        ts = df2.index.min()
        start_time_str = "{}-{:02d}-{:02d} 8:54:00".format(ts.year, ts.month, ts.day)
        start_ts = pd.Timestamp(start_time_str, tz=ts.tz)
        # periods=510 means from 9 to 17.29
        dti = pd.date_range(start_ts , periods=516, freq='min').to_series(keep_tz=True).rename('time')
        # remove from 17.25 - 17.28
        #dti.drop(dti.tail(5).head(4).index, inplace=True)
        df3 = df2.join(dti, how='right')
        if day_id == 0: # the first day, we must set the value from 8.55-8.59 as same as 9.00
            df3['last'].iloc[0] = df3['last'].iloc[5]
        else:
            df3['last'].iloc[0] = time_series_all_list[stock_id][day_id-1]['last'].iloc[-1]
        
        
        df3['last'].interpolate(method='linear', inplace=True)
        df3['volume'].iloc[:6] = df3['volume'].iloc[6] / 6
        df3['volume'].iloc[6] = df3['volume'].iloc[6] / 6
        df3['volume'].iloc[-5:] = df3['volume'].iloc[-1]/5
        
        
        df = df3.reset_index().rename({'index':'timestamp'}, axis=1)
        #df['timestamp'] = pd.to_datetime(df['time_stamp'], format="%Y-%m-%d %H:%M:%S").dt.tz_convert('Europe/Stockholm')
        df['ema_10'] = df['last'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['last'].ewm(span=20, adjust=False).mean()
        df['diff_ema_10']=(df['ema_10'].diff()[1:]/df['ema_10']).fillna(0)
        # the first diff at 9:00 is the difference between today's open and yesterday's last.

        df['diff_ema_20']=(df['ema_20'].diff()[1:]/df['ema_20']).fillna(0)
        df['value_ema_10_beta_99'] = 0
        df['value_ema_20_beta_99'] = 0
        df['value_ema_10_beta_98'] = 0
        df['value_ema_20_beta_98'] = 0
        for iter_id in range(10):
            df['value_ema_10_beta_99'] = df['diff_ema_10'].shift(-1).fillna(0) +                 0.99 * df['value_ema_10_beta_99'].shift(-1).fillna(0)
            df['value_ema_10_beta_98'] = df['diff_ema_10'].shift(-1).fillna(0) +                 0.98 * df['value_ema_10_beta_98'].shift(-1).fillna(0)
            df['value_ema_20_beta_99'] = df['diff_ema_20'].shift(-1).fillna(0) +                 0.99 * df['value_ema_20_beta_99'].shift(-1).fillna(0)
            df['value_ema_20_beta_98'] = df['diff_ema_20'].shift(-1).fillna(0) +                 0.98 * df['value_ema_20_beta_98'].shift(-1).fillna(0)
        # drop the first row because diff is nan    
        #df.drop(0, inplace=True)
        time_series_all_list[stock_id][day_id] = df.fillna(0)


# In[65]:


columns = ['timestamp','last', 'diff_ema_20', 'value_ema_20_beta_99', 'value_ema_20_beta_98', 
           'diff_ema_10', 'value_ema_10_beta_99', 'value_ema_10_beta_98', 'volume']


# In[70]:



from functools import reduce

# to order the columns in the order: last1,last2...last30, diff1, diff2,...diff30, value1, value2,...value30
def func(x):
    if 'timestamp' in x:
        return 0
    num = int(x.split('_')[-1])

    if 'last' in x:
        return 100 + num
    elif 'volume' in x:
        return 150 + num
    elif 'diff_ema_10' in x:
        return 200 + num
    elif 'diff_ema_20' in x:
        return 300 + num
    elif 'value_ema_10_beta_98' in x:
        return 400 + num
    elif 'value_ema_10_beta_99' in x:
        return 500 + num
    elif 'value_ema_20_beta_98' in x:
        return 600 + num
    elif 'value_ema_20_beta_99' in x:
        return 700 + num


def merge(df1, df2):
    global index
    merged = df1.set_index('timestamp').join(df2.set_index('timestamp'), how='outer')
    index += 1

    return merged.reset_index().rename({'index':'timestamp'}, axis=1)


columns_wanted = ['timestamp','last', 'diff_ema_20', 'value_ema_20_beta_99', 'value_ema_20_beta_98', 
           'diff_ema_10', 'value_ema_10_beta_99', 'value_ema_10_beta_98', 'volume']
df_day_list = []

for day_id in range(len(time_series_all_list[0])):
    print("processing day: {}".format(day_id))
    df_list = []
    for i in range(len(stock_id_list)):
        rename_map = {'last': 'last_' + str(i),
                      'diff_ema_20':'diff_ema_20_' + str(i),
                      'value_ema_20_beta_99':'value_ema_20_beta_99_' + str(i),
                      'value_ema_20_beta_98':'value_ema_20_beta_98_' + str(i),
                      'diff_ema_10':'diff_ema_10_' + str(i),
                      'value_ema_10_beta_99':'value_ema_10_beta_99_' + str(i),
                      'value_ema_10_beta_98':'value_ema_10_beta_98_'+ str(i),
                      'volume': 'volume_' + str(i)
                     }
        df = time_series_all_list[i][day_id][columns_wanted].rename(rename_map, axis=1)
        df_list.append(df)

    index = 0
    df_merged_daily = reduce(merge, df_list)
    columns = df_merged_daily.columns.tolist()
    columns.sort(key=func)
    df_merged_sorted = df_merged_daily[columns]
    
    df_day_list.append(df_merged_sorted)
    if day_id == 0:
        df_merged = df_merged_sorted
    else:
        df_merged = df_merged.append(df_merged_sorted)


# In[72]:


columns = df_merged_sorted.columns
last_columns = []
volume_columns = []
diff_ema_10_columns = []
diff_ema_20_columns = []
value_ema_10_beta_98_columns = []
value_ema_10_beta_99_columns = []
value_ema_20_beta_98_columns = []
value_ema_20_beta_99_columns = []
for item in columns:
    if 'last' in item:
        last_columns.append(item)
    elif 'volume' in item:
        volume_columns.append(item)
    elif 'diff_ema_10' in item:
        diff_ema_10_columns.append(item)
    elif 'diff_ema_20' in item:
        diff_ema_20_columns.append(item)
    elif 'value_ema_10_beta_98' in item:
        value_ema_10_beta_98_columns.append(item)
    elif 'value_ema_10_beta_99' in item:
        value_ema_10_beta_99_columns.append(item)
    elif 'value_ema_20_beta_98' in item:
        value_ema_20_beta_98_columns.append(item)
    elif 'value_ema_20_beta_99' in item:
        value_ema_20_beta_99_columns.append(item)


# In[73]:


df_merged[['timestamp'] + volume_columns + last_columns + diff_ema_10_columns + value_ema_10_beta_98_columns].to_csv('data-prep-ema10-beta98.csv')
df_merged[['timestamp'] + volume_columns + last_columns + diff_ema_10_columns + value_ema_10_beta_99_columns].to_csv('data-prep-ema10-beta99.csv')
df_merged[['timestamp'] + volume_columns + last_columns + diff_ema_20_columns + value_ema_20_beta_98_columns].to_csv('data-prep-ema20-beta98.csv')
df_merged[['timestamp'] + volume_columns + last_columns + diff_ema_20_columns + value_ema_20_beta_99_columns].to_csv('data-prep-ema20-beta99.csv')


# In[ ]:





# In[ ]:




