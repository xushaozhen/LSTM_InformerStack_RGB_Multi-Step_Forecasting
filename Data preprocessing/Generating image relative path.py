import os
import pandas as pd
import csv
import numpy as np
import sys
#构建文件的相对路径
#相对于当前工作目录或当前脚本文件所在目录的路径来指定文件或目录的位置
rgb_path = r"./img_data/rgb/"
#将相对路径转换为绝对路径
rgb_img_path = os.path.abspath(rgb_path)
rgb_img_list = os.listdir(rgb_img_path)
#print(rgb_img_list)
per_rgb_img_path_list = []
for rgb_img_name in rgb_img_list:
    per_rgb_img_name = rgb_img_path + '\\' + rgb_img_name
    #print(per_rgb_img_name)
    #'\\'反斜杠字符，用于在Windows文件路径中分隔目录和文件名
    per_rgb_img_path_list.append(per_rgb_img_name)

#print(per_rgb_img_path_list)

per_rgb_img_path_np = np.array(per_rgb_img_path_list)
#print(per_rgb_img_path_np)

per_rgb_img_path_dataframe = pd.DataFrame(per_rgb_img_path_np,index=None)
# print(per_rgb_img_path_dataframe)
per_rgb_img_path_dataframe.to_csv('per_rgb_img_path_relative.csv')










