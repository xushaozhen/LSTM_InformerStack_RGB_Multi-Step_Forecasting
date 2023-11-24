#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 00:06:28 2023

@author: xushaozhen
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 00:05:11 2023

@author: xushaozhen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 01:03:07 2023

@author: xushaozhen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 00:29:32 2023

@author: xushaozhen
"""
import time 
start = time.time()
import math
import os
import sys
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error #\u5e73\u65b9\u7edd\u5bf9\u8bef\u5dee
from sklearn.metrics import r2_score#R square
import shutil
import random
import copy

import os
import re
from tqdm import tqdm

# 设置随机种子为固定值
seed = 42  # 选择你想要的随机种子
batch_size=200
hidden_size=32
pred_len=24
epochs = 100
train_step = 0
n_hours = 96
n_features = 6
nhead=8
# 设置 PyTorch 的随机种子
torch.manual_seed(seed)

# 设置 Python 内置随机函数的种子
random.seed(seed)

# 设置 NumPy 库的随机函数的种子
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

Save_file_root_path="Only_LSTM_Multi_Steps"
results_save_path_root="./"+Save_file_root_path+"/"



# '''
# \u9996\u5148\u751f\u6210\u4e00\u4e2asin\u51fd\u6570\uff0c\u4f5c\u4e3a\u4f2a\u65f6\u95f4\u5e8f\u5217\u6570\u636e
# '''
# T = 1000
# x = torch.arange(1, T + 1, dtype=torch.float32)
# y = torch.sin(0.01 * x) + torch.normal(0, 0.1, (T,))#\u6bcf\u4e2ay\u52a0\u4e0a\u4e00\u4e2a0\u52300.2(\u5de6\u95ed\u53f3\u5f00)\u7684\u566a\u58f0
# # data = torch.arange(1, T + 1, dtype=torch.float32)
data = pd.read_csv("original_wait_find_2021_9_10_text_data.csv",header=0,index_col=0)
# print(data.head())
origin_values = data.values[:,:]
origin_values = origin_values.astype('float32')




# ratio = 0.67

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

# train_values = origin_values[:-365*24,:]
# # # print(train_values.shape)
# test_values = origin_values[-365*24:,:]
train_length=int(origin_values.shape[0]*0.7)
test_length=int(origin_values.shape[0]*0.2)
val_length=origin_values.shape[0]-train_length-test_length

train_values = origin_values[:train_length,:]
# # # print(train_values.shape)
val_values=origin_values[:train_length-n_hours,:]
test_values = origin_values[origin_values.shape[0]-test_length-n_hours:origin_values.shape[0],:]
# Test_y_value_reserves=test_values[:test_length-n_hours-(pred_len-1),-n_features]
# Test_y_value_reserves=test_values[:((test_values.shape[0]//batch_size)*batch_size-pred_len+1) ,-n_features]
# frame as supervised learning


reframed1 = series_to_supervised(train_values, n_hours, pred_len)
reframed2 = series_to_supervised(test_values, n_hours, pred_len)
# train_len = int(ratio*data.values.shape[0])

# train_values =  origin_values[:18379,:]
# # print(train_values.shape)
# test_values =  origin_values[-7876:,:]

scaler1 = MinMaxScaler(feature_range=(0,1))
scaled1 = scaler1.fit_transform(reframed1)

scaler2 = MinMaxScaler(feature_range=(0,1))
scaled2 = scaler2.fit_transform(reframed2)

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    time="%02d:%02d:%02d" % (h, m, s)
    return time

train = scaled1
test = scaled2
test1= copy.deepcopy(test)
test2= copy.deepcopy(test)
train_y_label=train[:,[i for i in range(-n_features*(pred_len-1)-1, 0, n_features)]]
test_y_label=test[:,[i for i in range(-n_features*(pred_len-1)-1, 0, n_features)]]
# split into train and test sets

# inv_x = test[:,-n_features+1:]
# print(inv_x.shape)

def dataset_split(train, test,train_y_label,test_y_label):
    print(train.shape)
    train_x = train[:,:n_hours*n_features]
    train_y = train_y_label
    test_x = test[:,:n_hours*n_features]
    test_y = test_y_label

    print(train_x.shape)
    train_x = train_x.reshape((train_x.shape[0],n_hours,n_features))
    test_x = test_x.reshape((test_x.shape[0],n_hours,n_features))

    print("train,test size:{},{},{},{}".format(train_x.shape,train_y.shape,test_x.shape,test_y.shape))
    return train_x,train_y,test_x,test_y
    # return train_features, train_target, test_features, test_target
train_x,train_y,test_x,test_y = dataset_split(train, test,train_y_label,test_y_label)



class dataset_prediction(Dataset):
    '''
    \u5c06\u4f20\u5165\u7684\u6570\u636e\u96c6\uff0c\u8f6c\u6210Dataset\u7c7b\uff0c\u65b9\u9762\u540e\u7eed\u8f6c\u5165Dataloader\u7c7b
    \u6ce8\u610f\u5b9a\u4e49\u65f6\u4f20\u5165\u7684data_features,data_target\u5fc5\u987b\u4e3anumpy\u6570\u7ec4
    '''
    def __init__(self, data_features, data_target):
        self.len = len(data_features)
        self.features = torch.from_numpy(data_features)
        self.target = torch.from_numpy(data_target)

    def __getitem__(self, index):
        return self.features[index], self.target[index]

    def __len__(self):
        return self.len

train_set = dataset_prediction(data_features=train_x, data_target=train_y)
test_set = dataset_prediction(data_features=test_x, data_target=test_y)

train_data_load = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True,drop_last=True)
test_data_load = DataLoader(dataset=test_set,batch_size=batch_size,shuffle=False,drop_last=True)

# print("lj8888888888888",len(train_set))
class Embeddings(nn.Module):
    #d_model\uff1a\u8bcd\u5d4c\u5165\u7eac\u5ea6
    #vocab: \u6b64\u8868\u7684\u5927\u5c0f
    def __init__(self,d_model,vocab):
        super(Embeddings, self).__init__()
        #\u5b9a\u4e49Embedding\u5c42
        self.lut = nn.Embedding(vocab,d_model)
        self.d_model = d_model
    def forward(self,x):
        #x\uff1a\u4ee3\u8868\u8f93\u5165\u8fdb\u6a21\u578b\u7684\u6587\u672c\u901a\u8fc7\u8bcd\u6c47\u6620\u5c04\u540e\u7684\u6570\u5b57\u5f20\u91cf
        return self.lut(x) * math.sqrt(self.d_model)


#\u6784\u5efa\u4f4d\u7f6e\u7f16\u7801\u5668\u7684\u7c7b
class PositionEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        # d_model\uff1a\u8bcd\u5d4c\u5165\u7eac\u5ea6
        #dropout \u7f6e\u96f6\u6bd4\u4f8b
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        #\u521d\u59cb\u5316\u4e00\u4e2a\u4f4d\u7f6e\u7f16\u7801\u77e9\u9635\uff0c\u5927\u5c0f\u662fmax_len * d_model
        pe = torch.zeros(max_len,d_model)
        #\u521d\u59cb\u5316\u7edd\u5bf9\u4f4d\u7f6e\u77e9\u9635
        position = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2) * -(math.log(10000.0))) / d_model

        #\u5c06\u524d\u9762\u5b9a\u4e49\u7684\u53d8\u5316\u77e9\u9635\u8fdb\u884c\u5947\u6570\uff0c\u5076\u6570\u7684\u5206\u522b\u8d4b\u503c
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        #\u5c06\u4e8c\u7ef4\u5f20\u91cf\u6269\u5145\u4e3a3\u7ef4
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        #x:\u6587\u672c\u5e8f\u5217\u7684\u8bcd\u5d4c\u5165\u8868\u793a
        x = x + Variable(self.pe[:,:x.size(1)],requires_grad = False)
        return self.dropout(x)

# hidden_size=64
#\u521b\u5efa\u6a21\u578b
class Mytransformer(nn.Module):
    def __init__(self,n_hours,n_features):
        super(Mytransformer, self).__init__()
        self.lstm = nn.LSTM(input_size=n_features,hidden_size=hidden_size,num_layers=1,
                          batch_first=True,)
        self.positionEncoding = PositionEncoding(hidden_size,0.2,max_len=100)
        self.transformer = nn.Transformer(d_model=hidden_size,nhead=nhead,num_encoder_layers=2,num_decoder_layers=2,dim_feedforward=64,dropout=0.3,activation='relu',)
        # self.lstm = nn.LSTM(4,64,num_layers=1)
        self.flatten = nn.Flatten()
        # self.linear = nn.Linear(64*4,1)
        self.linear = nn.Linear(hidden_size*n_hours,pred_len)

    def forward(self,x):
        lstm_out,_=self.lstm(x)
        #print(x.shape)
        # print("lstm_out.shape",lstm_out.shape)
        position_tgt = self.positionEncoding(lstm_out)
        # print("position_tgt.shape",position_tgt.shape)
        out = self.transformer(lstm_out,position_tgt)
        
        # print("transformer_out",out.shape)
        # out,_ = self.lstm(out)
        out = self.flatten(lstm_out)
        # print("flatten_out",out.shape)
        out = self.linear(out)
        out = nn.ReLU()(out)
        # out = out.view(out.shape[0],out.shape[1])
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = Mytransformer(n_hours,n_features)
model = model.to(device)

loss_fn = nn.MSELoss()
loss_fn = loss_fn.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loss = []
test_loss = []

best_test_loss = float('inf')  # 初始化最佳测试损失

for i in range(epochs):
    model.train()
    print("------------Epoch {}/{}------------".format(i+1,epochs))
    epoch_train_loss = 0.0
    for (x,y) in tqdm(train_data_load):
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        # print("out.shape",output.shape)
        loss = loss_fn(output,y)
        print()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()  # 累加每个batch的loss

    avg_train_loss = epoch_train_loss / len(train_set)  # 计算平均loss
    train_loss.append(avg_train_loss)  # 添加到训练loss列表中
    print("Average training loss:{}".format(avg_train_loss))
    print("End of batch {}, training loss:{}".format(i + 1, loss))

    pre = []
    total_loss = 0
    model.eval()
    with torch.no_grad():
        
        for (x,y) in tqdm(test_data_load):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
    
            loss = loss_fn(output, y)  # 先计算损失
            total_loss = total_loss + loss.item() 
            output = output.detach().cpu().numpy()
            # output = output.cpu()  # 然后将 output tensor 转移到 CPU
            # output_numpy = output.detach().numpy()  # 然后再将 tensor 转换成 numpy array
    
            pre.append(output)  # 使用转换后的 output_numpy
    # 使用转换后的 output_numpy
  # 使用转换后的 output_numpy

            
        avg_test_loss = total_loss / len(test_set)  # 计算平均loss
        test_loss.append(avg_test_loss)  # 添加到测试loss列表中
        print("Average test loss:{}".format(avg_test_loss))
        print("End of test {}, test total loss:{}".format(i + 1, loss))

    # 保存最好的模型
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        torch.save(model, 'best_model.pth')
        # print("New best model saved.")
        print("\033[1;32mNew best model saved\033[0m")
# 绘制训练和测试的loss曲线



# pre_values = np.array([])
pre_values = []
for i in range(len(pre)):
    pre_values.extend(pre[i])





# print(pre_values)
# for i in range(len(pre)):
#     pre_values = np.concatenate((pre_values,pre[i]))
List_of_test_set_labels=[i for i in range(-n_features*(pred_len-1)-1, 0, n_features)]
List_of_test_set_labels1 = [item for item in range(-pred_len, 0)]

pre_values=np.array(pre_values)
pre_values = pre_values.reshape((pre_values.shape[0], pred_len))
# test_y1=test_y
Test_y_value_reserves=reframed2.iloc[:pre_values.shape[0],List_of_test_set_labels]

test_y = test_y.reshape((test_y.shape[0], pred_len))

# train_=train[:,[i for i in range(-n_features*pred_len, 0, n_features)]]

#数据集标签列替换，方便反归一化
# test1[:,List_of_test_set_labels]=pre_values[:,List_of_test_set_labels1]
# index_row=((test_values.shape[0]//batch_size)*batch_size-pred_len+1)
test1[:pre_values.shape[0],List_of_test_set_labels]=pre_values[:pre_values.shape[0],List_of_test_set_labels1]
# inv_pre = np.concatenate((pre_values,inv_x[:,:]),axis=1)
# inv_test_y = np.concatenate((test_y,inv_x[:,:]),axis=1)
# print(inv_pre.shape)
# print(inv_test_y.shape)

last_pre_values1 = scaler2.inverse_transform(test1)
last_pre_values=last_pre_values1[:pre_values.shape[0],List_of_test_set_labels]

#取出测试的真实值
# test_real_value_1 = scaler2.inverse_transform(test2)
# # final_pre_value=last_pre_values[:,-n_features]
# test_real_value=test_real_value_1[:,List_of_test_set_labels]
test_real_value=np.array(Test_y_value_reserves).reshape(-1,pred_len)
# for i in range(len(test_y[:,-n_features])):
#     print(last_pre_values[i,-n_features],test_y[i,-n_features])
# rmse = np.sqrt(mean_squared_error(test_real_value,final_pre_value))
# print("rmse:{}".format(rmse))

# rmse1 = np.sqrt(mean_squared_error(test_real_value,final_pre_value))
# nrmse1=rmse1/(np.mean(test_real_value,axis=0))
# mae1=mean_absolute_error(test_real_value,final_pre_value)
# R21=r2_score(test_real_value,final_pre_value)


# print("\033[1;32m******main_output\u53c2\u6570\u8bc4\u4f30\u7ed3\u679c******\033[0m")
# print('Output_test_RMSE: %.3f' % rmse1)
# print('Output_test_nrmse: %.3f' % nrmse1)
# print('Output_test_mae: %.3f' % mae1)
# print('Output_test_R^2: %.3f' % R21)

# print("\033[1;32m********************************************\033[0m")


# Dataframe_Final_Evaluation_Index = pd.DataFrame(data={'RMSE': [rmse1], 'nRMSE': [nrmse1], 'MAE': [mae1], 'R^2': [R21]})
# Dataframe_Final_Evaluation_Index.to_csv("Final_Evaluation_Index_Statistics_Results.csv", index=None)

# Dataframe_Final_Evaluation_Pre_Real_values=pd.DataFrame(data={'measured value':np.array(test_real_value).reshape(-1,),'predicted value':np.array(final_pre_value).reshape(-1,)})
# Dataframe_Final_Evaluation_Pre_Real_values.to_csv("Final_Evaluation_Pre_Real_values.csv",index=None)
# results_save_path_root="./Only_LSTM/"



# files_in_folder=os.listdir(results_save_path_root)

# for files_in_folder_path in files_in_folder:
#         temp_split_path=files_in_folder_path.split("_")[-1]
#             files_in_folder=os.listdir(folder[folder_number])

def retain_min_folder(folder_path):
    # 获取文件夹列表
    folder_list = os.listdir(folder_path)

    # 提取数字部分并转换为浮点数
    pattern = r'\d+\.\d+'
    folder_numbers = [float(re.search(pattern, folder_name).group()) for folder_name in folder_list if re.search(pattern, folder_name)]

    if folder_numbers:
        # 找到数字最小的文件夹
        min_folder_number = min(folder_numbers)
        min_folder_name = f'Only_LSTM_{min_folder_number}'

        for folder_name in folder_list:
            # 检查文件夹名称是否匹配并删除其子文件和文件夹
            if folder_name.startswith('Only_LSTM_') and folder_name != min_folder_name:
                folder_path = os.path.join(folder_path, folder_name)
                shutil.rmtree(folder_path)
                print(f'Deleted folder: {folder_path}')

        print(f'Retained folder: {min_folder_name}')

    else:
        print('No matching folders found in the specified directory.')


def model_files_automatically_deleted(final_results_save_path):
     
    if os.path.exists(final_results_save_path):
        print("\033[1;32m*****folder already exists******\033[0m")
        # shutil.rmtree(final_results_save_path)
    else:
        os.makedirs(final_results_save_path)
def Save_current_script_to_specified_directory(final_results_save_path):  #将每个单词用下划线链接输出
    if sys.platform.startswith('win'):
        script_name = sys.argv[0].split("\\")[-1]   
        print("The current operating system is Windows")
    elif sys.platform.startswith('linux'):
        script_name = sys.argv[0].split("/")[-1] 
        print("The current operating system is Linux")
    else:
        print("The current operating system is another OS")

    destination_folder = final_results_save_path
    
    destination_file_path = os.path.join(destination_folder, script_name)
    
    shutil.copy(sys.argv[0], destination_file_path)
        
# model_files_automatically_deleted(final_results_save_path)
# Save_data_to_a_specified_file(final_results_save_path,Dataframe_Final_Evaluation_Index,Dataframe_Final_Evaluation_Pre_Real_values,model,plt)
# folder_path = '/path/to/folder'  # 替换为实际的文件夹路径

# 调用函数
# retain_min_folder(results_save_path_root)


# def plot_train_test_loss_and_save(train_losses_list,vali_loss_list,test_losses_list,save_path):
#     plt.plot(train_loss_list, label='Training loss')
#     plt.plot(vali_loss_list, label='Vali loss')
#     plt.plot(test_loss_list, label='Test loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig(save_path+"/loss_plot.png")
#     plt.show()


def calculate_rmse(predictions, targets):
     
    if len(predictions) != len(targets):
        raise ValueError("The shape of the predicted and true values do not match.")
    
    # 将数组转换为 NumPy 数组
    predictions = np.array(predictions)
    targets = np.array(targets)

    # 计算预测值与真实值之间的差值
    errors = predictions - targets

    # 计算均方根误差
    rmse = np.sqrt(np.mean(errors**2))

    return rmse


def calculate_multi_step_all_samples_rmse(predictions, targets):
     
    if predictions.shape != targets.shape:
        raise ValueError("The shape of the predicted and true values do not match.")
    
    num_steps = predictions.shape[1]
    num_samples = predictions.shape[0]
    rmse_values_list = []

    for i in range(num_steps):
        prediction_step = predictions[:, i]
        target_step = targets[:, i]

        rmse_step = calculate_rmse(prediction_step, target_step)
        rmse_values_list.append(rmse_step)

    return rmse_values_list
def RMSE_after_flattening_of_all_samples(predictions, targets):
    predictions_1=np.array(predictions).reshape(1,-1)
    predictions=predictions_1.reshape(-1,1)
   
    targets_1=np.array(targets).reshape(1,-1)
    
    targets=targets_1.reshape(-1,1)
    print('predictions &targets shape:', predictions.shape, targets.shape)
    all_samples_flattening_rmse=calculate_rmse(predictions, targets)
    return all_samples_flattening_rmse
def save_trues_pred(trues,pred,step_count,save_path_results_name):
    combined_data = np.concatenate((trues, pred), axis=1)
    
    # 生成列名（假设每个步长都有10列）
     
    column_names = [f"true_step{i+1}" for i in range(step_count)] + [f"pred_step{i+1}" for i in range(step_count)]
    
    # 创建DataFrame，并使用列名
    df = pd.DataFrame(combined_data, columns=column_names)
    
    # CSV文件名
    csv_file = save_path_results_name+".csv"
    
    # 将DataFrame写入CSV文件
    df.to_csv(csv_file, index=True)
def write_to_csv(single_value, my_list, csv_file):
    # 将列表转换为DataFrame，并将每个元素作为一行
    df_list = pd.DataFrame({'M_step_rmse': my_list})

    # 创建包含单个值的DataFrame
    df_single = pd.DataFrame({'all_samples_flattening_rmse': [single_value]})

    # 将单个值和列表的DataFrame进行合并，并将其写入CSV文件
    df_combined = pd.concat([df_single, df_list], ignore_index=True)
    df_combined.to_csv(csv_file, index=False)


targets=test_real_value
predictions=last_pre_values
all_samples_flattening_rmse=RMSE_after_flattening_of_all_samples(predictions, targets)
rmse_values_list=calculate_multi_step_all_samples_rmse(predictions, targets)

#final_results_save_path="./Only_LSTM/Only_LSTM_"+str(rmse1)

# rmse1 = np.sqrt(mean_squared_error(test_real_value,final_pre_value))
# nrmse1=rmse1/(np.mean(test_real_value,axis=0))
# mae1=mean_absolute_error(test_real_value,final_pre_value)
# R21=r2_score(test_real_value,final_pre_value)

final_results_save_path="./"+Save_file_root_path+"/"+Save_file_root_path+"_"+str(all_samples_flattening_rmse)



def plot_train_test_loss_and_save(train_loss,test_loss,final_results_save_path):
    plt.figure()
    plt.plot(train_loss, label='Training loss')
    plt.plot(test_loss, label='Test loss')
    plt.xlabel('Epoch',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.legend(fontsize=14)
    plt.savefig(final_results_save_path+"/loss_plot.png")
    plt.show()

def Save_data_to_a_specified_file(train_loss,test_loss,final_results_save_path,rmse_values_list,model,all_samples_flattening_rmse):
      
    plot_train_test_loss_and_save(train_loss,test_loss,final_results_save_path)
    write_to_csv(all_samples_flattening_rmse, rmse_values_list, final_results_save_path+"/Final_Evaluation_Index_Statistics_Results.csv",)
    save_trues_pred(targets,predictions,step_count=len(rmse_values_list),save_path_results_name=final_results_save_path+"/Final_Evaluation_Pre_Real_values")
   
    torch.save(model,final_results_save_path+"/best_model_"+str(all_samples_flattening_rmse )+".pth")
    print("\033[1;32mCongratulations! The model file has been deleted. You are loading the dataset for training\033[0m")   
    Save_current_script_to_specified_directory(final_results_save_path)



print("\033[1;32m******main_output\u53c2\u6570\u8bc4\u4f30\u7ed3\u679c******\033[0m")



print('all_samples_flattening_rmse: %.3f' % all_samples_flattening_rmse)
# print('step1_RMSE: %.3f' % rmse_values_list[0])
# print('step2_RMSE: %.3f' % rmse_values_list[1])
# # print('step3_RMSE: %.3f' % rmse_values_list[2])
# # print('step4_RMSE: %.3f' % rmse_values_list[3])
for step, rmse in enumerate(rmse_values_list):
    print(f"step{step+1}_RMSE: {rmse}")
print("\033[1;32m******main_output\u53c2\u6570\u8bc4\u4f30\u7ed3\u679c******\033[0m")

def plot_multi_steps_rmse(multi_steps_rmse_list,save_path):
    plt.figure()
    plt.plot(multi_steps_rmse_list,color='r')
    
    plt.xlabel('step',fontsize=16)
    plt.ylabel('rmse',fontsize=16)
    plt.yticks( fontsize=14)
    plt.title('Multi-step prediction of RMSE curves',fontsize=16)
    plt.autoscale()
    plt.xticks([item for item in range(len(multi_steps_rmse_list))], [str(item+1) for item in range(len(multi_steps_rmse_list))],fontsize=14)
    plt.legend()
    plt.savefig(save_path+"/Multi-step prediction of RMSE curves.png",dpi=600)
    plt.show()




# Dataframe_Final_Evaluation_Index = pd.DataFrame(data={'All_samples_flattening_rmse': [all_samples_flattening_rmse],'Step_rmse':np.array(rmse_values_list).reshape(-1, 1)})
# Dataframe_Final_Evaluation_Index = pd.DataFrame(data={'Step_rmse':np.array(rmse_values_list).reshape(-1, 1)})
# Dataframe_Final_Evaluation_Index.to_csv("Final_Evaluation_Index_Statistics_Results.csv", index=None)

# Dataframe_Final_Evaluation_Pre_Real_values=pd.DataFrame(data={'measured value':np.array(targets),'predicted value':np.array(predictions)})
# Dataframe_Final_Evaluation_Pre_Real_values.to_csv("Final_Evaluation_Pre_Real_values.csv",index=None)


model_files_automatically_deleted(final_results_save_path)
Save_data_to_a_specified_file(train_loss,test_loss,final_results_save_path,rmse_values_list,model,all_samples_flattening_rmse)
# retain_min_folder(results_save_path_root)

plot_multi_steps_rmse(rmse_values_list,final_results_save_path)

# Save_data_to_a_specified_file(final_results_save_path,Dataframe_Final_Evaluation_Index,Dataframe_Final_Evaluation_Pre_Real_values,model,plt)
# retain_min_folder(results_save_path_root)
 
end = time.time()
time=format_time(end-start)

print("\033[1;32m??????:\033[0m",time)
torch.cuda.empty_cache()




print("\033[1;32m\u7a0b\u5e8f\u6267\u884c\u65f6\u95f4:\033[0m",time)
# print("\u7a0b\u5e8f\u6267\u884c\u65f6\u95f4:",time) 