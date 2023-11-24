# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 22:31:28 2023

@author: Liu_Jun_Desktop
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
import matplotlib.pyplot as pltmetric
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error #??????
from sklearn.metrics import r2_score#R square
import shutil
import random    
# from models.model import Informer, InformerStack
# from models.model_lstm_concat_add_features_weights_add_Customized_3DConv import Informer, InformerStack
# from utils.tools import EarlyStopping, adjust_learning_rate
# from utils.metrics import metric
# from utils.timefeatures import time_features 
# from utils.tools import StandardScaler
# from utils.Data_Deletion_Cleaning_Matching import Data_deletion,DataLookupAndMatch
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
# from models.Informer import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score
import os
import time
import matplotlib.pyplot as plt
import warnings
import re
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import transforms
# torch.cuda.empty_cache()

train_epochs = 100
batch_size=2
freq='t'
padding=0
#automatic mixed precision 自动混合精度，一种深度学习技巧旨在加速训练并降低内存损耗
use_amp = False
output_attention = False
inverse = True
n_features,enc_in,dec_in=6,6,6
c_out=1
n_hours,seq_len=24,24
label_len=12
pred_len,out_len=1,1
input_channels,height,width=3,32,32
images_features=1
d_model=256 
n_heads=8
# e_layers=2 
d_layers=2
d_ff=2048
dropout=0
learning_rate=0.001
patience=7
Save_file_root_path="InformerStack_Train_test_multi_step_ourdatasets"
results_save_path_root="./"+Save_file_root_path+"/"
seed=47
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

text_data_path='original_wait_find_2021_9_10_text_data.csv'
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
class EvaluationMetrics:
    def __init__(self):
        # You can add any necessary initializations here
        pass
    def rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def nmae(self, y_true, y_pred):
        #return np.mean(np.abs(y_pred-y_true)/ y_true)
        return mean_absolute_error(y_true, y_pred)/ (np.max(y_true)-np.min(y_true))
        #return mean_absolute_error(y_true, y_pred)/ np.mean(y_true, axis=0)
   
    def calculate_rmse(self,predictions, targets):
         
        if len(predictions) != len(targets):
            raise ValueError("The shape of the predicted and true values do not match.")
        
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        errors = predictions - targets
        rmse = np.sqrt(np.mean(errors**2))

        return rmse

    def nrmse(self, y_true, y_pred):
        rmse2 = self.rmse(y_true, y_pred)
        return rmse2 / np.mean(y_true, axis=0)

    def mape(self, actual, pred):
        actual, pred = np.array(actual), np.array(pred)
        return np.mean(np.abs((actual - pred) / actual))

    def smape(self, y_true, y_pred):
        return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

    def R2(self, y_true, y_pred):
        return r2_score(y_true, y_pred)

    def _error(self, actual, predicted):
        """ Simple error """
        return actual - predicted

    def _percentage_error(self, actual, predicted):
        """ Percentage error """
        return (actual - predicted) / actual

    def medape(self, actual, predicted):
        """
        Median Absolute Percentage Error
        Note: result is NOT multiplied by 100
        """
        return np.median(np.abs(self._percentage_error(actual, predicted)))

    def FS_RMSE(self, model_y_true, model_y_pred, persistence_rmse):
        model_rmse = self.rmse(model_y_true, model_y_pred)
        persistence_rmse = persistence_rmse
        return 1 - (model_rmse / persistence_rmse)

    def FS_nRMSE(self, model_y_true, model_y_pred, persistence_nrmse):
        model_nrmse = self.nrmse(model_y_true, model_y_pred)
        persistence_nrmse = persistence_nrmse
        return 1 - (model_nrmse / persistence_nrmse)
    
    def calculate_multi_step_all_samples_rmse(self,predictions, targets):
         
        if predictions.shape != targets.shape:
            raise ValueError("The shape of the predicted and true values do not match.")
        
        num_steps = predictions.shape[1]
        num_samples = predictions.shape[0]
        rmse_values_list = []
        nrmse_values_list=[]
        mape_values_list=[]
        smape_values_list=[]
        R2_values_list=[]
        
        for i in range(num_steps):
            prediction_step = predictions[:, i]
            target_step = targets[:, i]
           
            rmse_step = self.calculate_rmse(prediction_step, target_step)
            nrmse_step=self.nrmse(target_step,prediction_step)
            mape_step=self.medape(target_step,prediction_step)
            smape_step=self.smape(target_step,prediction_step)
            # smape_step=1-(mape_step/persistence_nrmse)
            #print("smape_step:",smape_step)
            R2_step=self.R2(target_step,prediction_step)
             
            rmse_values_list.append(rmse_step)
            nrmse_values_list.append(nrmse_step)
            mape_values_list.append(mape_step)
            smape_values_list.append(smape_step)
            R2_values_list.append(R2_step)
            #print("smape_values_list:",smape_values_list)
        all_samples_nrmse_mean=np.mean(nrmse_values_list)
        all_samples_mape_mean=np.mean(mape_values_list)
        all_samples_smape_mean=np.mean(smape_values_list)
        all_samples_R2_mean=np.mean(R2_values_list)

        return [rmse_values_list,nrmse_values_list,mape_values_list,
            smape_values_list,R2_values_list,all_samples_nrmse_mean,
            all_samples_mape_mean,all_samples_smape_mean,
            all_samples_R2_mean]
        
    
    def RMSE_after_flattening_of_all_samples(self,predictions, targets):
        predictions_1=np.array(predictions).reshape(1,-1)
        predictions=predictions_1.reshape(-1,1)
       
        targets_1=np.array(targets).reshape(1,-1)
        
        targets=targets_1.reshape(-1,1)
        print('predictions &targets shape:', predictions.shape, targets.shape)
        all_samples_flattening_rmse=self.calculate_rmse(predictions, targets)
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
import csv
def write_metrics_to_csv(filename, rmse, nrmse, mape, smape, r2, extra_data):
    def csv_exists(filename):
        return os.path.exists(filename) and os.path.isfile(filename)
    
    mode = 'w'
    if csv_exists(filename):
        mode = 'w'
    
    with open(filename, mode=mode, newline='') as file:
        writer = csv.writer(file)
        if mode == 'w':
            writer.writerow(["RMSE", "nRMSE", "MAPE", "sMAPE", "R2"])  # 写入标题行
            for vals in zip(rmse, nrmse, mape, smape, r2):
                writer.writerow(vals)  # 逐列写入数据
        writer.writerow(["", "", "", "", "", "Multi-step prediction of the mean value of each indicator"])  # 写入一个空行和标题行
        writer.writerow(["", "", "", "", "", ""])  # 写入空行
        writer.writerow(["", "", "", "", "", ""]) 
        writer.writerow(["all_samples_rmse", "nrmse_mean", "mape_mean", "sampe_mean", "R2_mean"]) 
        writer.writerow(extra_data)  # 写入追加的数据
        
    print("数据已写入", filename)



# 调用函数


data = pd.read_csv(text_data_path,header=0,index_col=0)
origin_values = data.values[:,:]
origin_values = origin_values.astype('float32')
 
train_length=int(origin_values.shape[0]*0.6)
test_length=int(origin_values.shape[0]*0.2)
val_length=origin_values.shape[0]-train_length-test_length

train_values = origin_values[:train_length,:]
val_values=origin_values[train_length-n_hours:train_length+val_length,:]
test_values = origin_values[origin_values.shape[0]-test_length-n_hours:origin_values.shape[0],:]

reframed1 = series_to_supervised(train_values, n_hours, pred_len)
reframed2 = series_to_supervised(test_values, n_hours, pred_len)


# future_pred_ghi_list_index=-pred_len*n_features-1
future_pred_csi_list_index=-pred_len*n_features-3
future_real_ghi_list=[i for i in range(-n_features*(pred_len-1)-1, 0, n_features)]
future_theroy_ghi_list_index=[i-1 for i in range(-n_features*(pred_len-1)-1, 0, n_features)]
# List_of_load_set = [item for item in range(-pred_len, 0)]

future_pred_csi_array = np.expand_dims(np.array(reframed2.iloc[:reframed2.shape[0],future_pred_csi_list_index]), axis=1)
future_real_theroy_ghi=reframed2.iloc[:reframed2.shape[0],future_theroy_ghi_list_index]
# 在第二个维度上重复广播的列数次
future_repeat_csi = np.repeat(future_pred_csi_array, len(future_real_ghi_list), axis=1)
future_real_theroy_ghi=reframed2.iloc[:reframed2.shape[0],future_theroy_ghi_list_index]
#应csi乘以理论ghi得出预测ghi
future_pred_ghi =future_repeat_csi*future_real_theroy_ghi

 
# future_pred_ghi[:reframed2.shape[0],:len(future_real_ghi_list)]=np.array(reframed2.iloc[:reframed2.shape[0],future_pred_ghi_list_index])
future_real_ghi=reframed2.iloc[:reframed2.shape[0],future_real_ghi_list]


transfer_future_pred_ghi=np.array(future_pred_ghi).reshape(-1,pred_len)
transfer_future_real_ghi=np.array(future_real_ghi).reshape(-1,pred_len)
 

# transfer_future_pred_ghi,transfer_future_real_ghi 



 
# transfer_future_pred_ghi,transfer_future_real_ghi =get_test_real_ghi_theroy_clean_ghi_csi(n_hours,pred_len,n_features)
# rea_csi,theory_ghi,real_ghi=get_test_real_ghi_theroy_clean_ghi_csi(csi_preds,n_hours,pred_len,n_features)  
# pred_ghi=csi_preds*theory_ghi

targets=transfer_future_real_ghi
predictions=transfer_future_pred_ghi
eval_metrics = EvaluationMetrics()
all_samples_flattening_rmse=eval_metrics.RMSE_after_flattening_of_all_samples(predictions, targets)

[rmse_values_list,nrmse_values_list,mape_values_list,
    smape_values_list,R2_values_list,all_samples_nrmse_mean,
    all_samples_mape_mean,all_samples_smape_mean,
    all_samples_R2_mean] = eval_metrics.calculate_multi_step_all_samples_rmse(predictions, targets)

results_list=[rmse_values_list,nrmse_values_list,mape_values_list,
    smape_values_list,R2_values_list,all_samples_nrmse_mean,
    all_samples_mape_mean,all_samples_smape_mean,
    all_samples_R2_mean]

extra_data_mean=[all_samples_flattening_rmse,all_samples_nrmse_mean,
                 all_samples_mape_mean,all_samples_smape_mean,all_samples_R2_mean]

# rmse1 = np.sqrt(mean_squared_error(test_real_value,final_pre_value))
# nrmse1=rmse1/(np.mean(test_real_value,axis=0))
# mae1=mean_absolute_error(test_real_value,final_pre_value)
# R21=r2_score(test_real_value,final_pre_value)

final_results_save_path="./"+Save_file_root_path+"/"+Save_file_root_path+"_"+str(all_samples_flattening_rmse)

 




print("\033[1;32m******main_output\u53c2\u6570\u8bc4\u4f30\u7ed3\u679c******\033[0m")



print('all_samples_flattening_rmse: %.3f' % all_samples_flattening_rmse)
# print('step1_RMSE: %.3f' % rmse_values_list[0])
# print('step2_RMSE: %.3f' % rmse_values_list[1])
# # print('step3_RMSE: %.3f' % rmse_values_list[2])
# # print('step4_RMSE: %.3f' % rmse_values_list[3])
for step, rmse in enumerate(rmse_values_list):
    print(f"step{step+1}_RMSE: {rmse}")
 

print("*" * 25)
print('all_samples_rmse_mean: %.3f' % np.mean(results_list[0]))
print('nRMSE_mean: %.3f' % all_samples_nrmse_mean)
print('medape_mean: %.3f' % all_samples_mape_mean)
print('smape_mean: %.3f' % all_samples_smape_mean)
print('R2_mean: %.3f' % all_samples_R2_mean)
print("************step_nrmse,srep_medape,step_smape,step_r2**************" )
print("step_nrmse:\n", results_list[1] )
print("step_medape:\n", results_list[2] )
print("step_smape:\n", results_list[3] )
print("step_R2:\n", results_list[4] )


print('nRMSE_mean: %.3f' % all_samples_nrmse_mean)
print("\033[1;32m******main_output\u53c2\u6570\u8bc4\u4f30\u7ed3\u679c******\033[0m")



def plot_multi_steps_rmse(multi_steps_rmse_list,save_path):
    plt.figure()
    plt.plot(multi_steps_rmse_list,color='r')
    
    plt.xlabel('step',fontsize=16)
    plt.ylabel('rmse',fontsize=16)
    plt.yticks( fontsize=16)
    plt.title('Multi-step prediction of RMSE curves',fontsize=16)
    #RMSE曲线的多步预测
    plt.autoscale()
    if pred_len<=12:
      plt.xticks([item for item in range(len(multi_steps_rmse_list))], [str(item+1) for item in range(len(multi_steps_rmse_list))],fontsize=14)  
    elif 12<pred_len<=48:
       plt.xticks([item for item in range(0,len(multi_steps_rmse_list),4)], [str(item+1) for item in range(0,len(multi_steps_rmse_list),4)],fontsize=14)  
    elif 48<pred_len<=96:
       plt.xticks([item for item in range(0,len(multi_steps_rmse_list),6)], [str(item+1) for item in range(0,len(multi_steps_rmse_list),6)],fontsize=14)    
    elif 96<pred_len<=168:
       plt.xticks([item for item in range(0,len(multi_steps_rmse_list),8)], [str(item+1) for item in range(0,len(multi_steps_rmse_list),9)],fontsize=12)
    else:
       plt.xticks([item for item in range(0,len(multi_steps_rmse_list),24)], [str(item+1) for item in range(0,len(multi_steps_rmse_list),24)],fontsize=10)
    plt.legend(fontsize=14)
    # plt.savefig(save_path+"/Multi-step prediction of RMSE curves.png",dpi=600)
    plt.show()

 

# Dataframe_Final_Evaluation_Index = pd.DataFrame(data={'All_samples_flattening_rmse': [all_samples_flattening_rmse],'Step_rmse':np.array(rmse_values_list).reshape(-1, 1)})
# Dataframe_Final_Evaluation_Index = pd.DataFrame(data={'Step_rmse':np.array(rmse_values_list).reshape(-1, 1)})
# Dataframe_Final_Evaluation_Index.to_csv("Final_Evaluation_Index_Statistics_Results.csv", index=None)

# Dataframe_Final_Evaluation_Pre_Real_values=pd.DataFrame(data={'measured value':np.array(targets),'predicted value':np.array(predictions)})
# Dataframe_Final_Evaluation_Pre_Real_values.to_csv("Final_Evaluation_Pre_Real_values.csv",index=None)

write_metrics_to_csv("./Final_Evaluation_Index_Statistics_Results.csv", results_list[0],results_list[1],results_list[2],results_list[3],results_list[4], extra_data_mean)
save_trues_pred(targets,predictions,pred_len,"./Final_Pred_trues_Statistics_Results.csv")
# Save_data_to_a_specified_file(train_loss_list,vali_loss_list,test_loss_list,
#                                   final_results_save_path,rmse_values_list,model,all_samples_flattening_rmse)
plot_multi_steps_rmse(nrmse_values_list,final_results_save_path)

# Save_data_to_a_specified_file(final_results_save_path,Dataframe_Final_Evaluation_Index,Dataframe_Final_Evaluation_Pre_Real_values,model,plt)
# retain_min_folder(results_save_path_root)
 
# end = time.time()
# time=format_time(end-start)

# print("\033[1;32m Total code running time:\033[0m",time)
torch.cuda.empty_cache()

