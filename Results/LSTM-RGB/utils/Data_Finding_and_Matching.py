# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 03:04:06 2023

@author: Liu_Jun_Desktop
"""

import pandas as pd

# 从第一个CSV文件的路径中提取时间戳
def extract_timestamp(path):
    timestamp = path.split('_')[2] + "_" + path.split('_')[3]
    return timestamp

# 将时间戳转换为第二个CSV文件的时间格式
def convert_timestamp(timestamp):
    date, time = timestamp.split('_')
    formatted_date = f"{date[:4]}/{int(date[4:6])}/{int(date[6:8])} {int(time[:2])}:{time[2:4]}"
    return formatted_date

# 读取第一个CSV文件
df1 = pd.read_csv('2021_9_10_images_path.csv', header=0, index_col=0)

# 从每行数据中提取时间戳，并转换为第二个CSV文件的时间格式
df1['date'] = df1['rgb'].apply(lambda x: convert_timestamp(extract_timestamp(x)))

# 读取第二个CSV文件
df2 = pd.read_csv('wait_find_2021_9_10_text_data.csv')

# 使用时间戳从第一个CSV文件查找第二个CSV文件中的匹配数据
result_df = df1.merge(df2, on='date', how='inner')  # 这里使用'inner'确保只保留匹配的行

# 只保留所需的数据列
final_df = result_df[['date', 'Temp', 'Hum', 'Height', 'csi', 'Theory', 'ghi']]

# 将结果写入新的CSV文件，其中索引为时间戳
final_df.set_index('date').to_csv('output.csv')

# 检查新的CSV（即final_df）的行数与第一个CSV（即df1）的行数是否相等
if len(final_df) == len(df1):
    print("The new CSV has the same number of rows as the first CSV.")
else:
    print(f"The new CSV has {len(final_df)} rows while the first CSV has {len(df1)} rows.")
