# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 16:08:25 2023

@author: Liu_Jun_Desktop
"""
import pandas as pd
import numpy as np
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
# def get_test_real_ghi_theroy_clean_ghi_csi(pre_values,n_hours,pred_len,n_features):
#     data = pd.read_csv(text_data_path,header=0,index_col=0)
#     origin_values = data.values[:,:]
#     origin_values = origin_values.astype('float32')
     



# 读取CSV文件
df = pd.read_csv('original_2021_9_10_images_path.csv',header=0,index_col=0)

dataframe=series_to_supervised(df, n_in=24, n_out=1, dropnan=True)
# 首先保存原始数据到另一个CSV文件
# dataframe.to_csv('original_file.csv', index=True)

# 函数，从路径中提取日期
def extract_date(path):
    # 根据你的描述，从数据中提取日期
    date_part = path.split('_')[2]
    return date_part[:8]  # 取日期的部分

# 用来存储要删除的行号
rows_to_drop = []

# 遍历每一行
for index, row in dataframe.iterrows():
    # 提取该行所有数据的日期
    dates = [extract_date(str(cell)) for cell in row if isinstance(cell, str) and '_CAM1_' in cell]
    
    # 如果所有日期不相同，将行号添加到rows_to_drop中
    if len(set(dates)) > 1:
        rows_to_drop.append(index)
 
# 删除指定的行
dataframe.drop(rows_to_drop, inplace=True)

# print(rows_to_drop)
# 将结果写回CSV
#dataframe.to_csv('modified_file.csv', index=False)
# print(f"Deleted rows: {', '.join(map(str, rows_to_drop))}")
n_featues=1

sub_series_a=dataframe.iloc[:,:n_featues]
sub_series_b=np.array(dataframe.iloc[-1,n_featues:]).reshape(-1,n_featues)


result = np.concatenate((sub_series_a, sub_series_b),axis=0)
cleaning_data=pd.DataFrame(result)

cleaning_data.columns = ['rgb']

cleaning_data.to_csv('2021_9_10_images_path.csv', index=True)
# 打印被删除的行号




    