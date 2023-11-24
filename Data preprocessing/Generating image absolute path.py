import os
import pandas as pd
import csv
import numpy as np
#构建文件的绝对路径
#从文件系统的根目录开始描述文件或目录的完整路径
#r 将字符串中的反斜杠视为普通字符，而不进行转义
rgb_path = r"/img_data/rgb/"
#获取rgb图像文件的完整路径  os.getcwd() 获取到的当前工作目录与 rgb_path 变量拼接
rgb_img_path = os.getcwd() + rgb_path
rgb_img_list = os.listdir(rgb_img_path)

# print(rgb_img_path)
# print(rgb_img_list)
#使用PIL(python imaging library)库的功能处理图像文件的创建、编辑和处理
from PIL import Image

per_rgb_img_path_list = []
for rgb_img_name in rgb_img_list:
    per_rgb_img_name = rgb_img_path + rgb_img_name
    #print(per_rgb_img_name)
    per_rgb_img_path_list.append(per_rgb_img_name)
    #print(per_rgb_img_path_list)
im = Image.open(per_rgb_img_name)
im.show()


#将列表转换为numpy数组
per_rgb_img_path_np = np.array(per_rgb_img_path_list)
print(per_rgb_img_path_np)
#将numpy数组转换为Pandas DataFrame,并保存到csv文件中
per_rgb_img_path_dataframe = pd.DataFrame(per_rgb_img_path_np,index=None)
#print(per_rgb_img_path_dataframe)
per_rgb_img_path_dataframe.to_csv('per_rgb_img_path.csv')
#(128,126)宽为128，高为126，水平方向上有128个像素，垂直方向上有126个像素
#24,24,24 RGB三通道红、绿、蓝的值



