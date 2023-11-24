import pandas as pd
import numpy as np
#index_col=0 将文件中的第一列作为索引（行标签），header=0 第一行包含列名
data = pd.read_csv('supervised_per_rgb_img_path_relative-no-17hour-8hour.csv',header=0,index_col=0)
#print(data)
path_list = []
for i in range(len(data)):
    if data.values[i,-1].split("_")[-2] not in ["080000","080500","081000","081500","082000","082500"]:
        print(data.values[i,-1])
        path_list.append(data.values[i,:])
print(path_list)
#index=None 不指定索引列
per_rgb_img_path_dataframe = pd.DataFrame(path_list,index=None)
print(per_rgb_img_path_dataframe)
per_rgb_img_path_dataframe.to_csv('supervise_per_rgb_img.csv')
