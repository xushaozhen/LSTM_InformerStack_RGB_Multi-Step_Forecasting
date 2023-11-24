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
 

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
# from models.Informer import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error #\u5e73\u65b9\u7edd\u5bf9\u8bef\u5dee
from sklearn.metrics import r2_score#R square
import os
 
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
use_amp = False
output_attention = False
inverse = True

n_features,enc_in,dec_in=6,6,6
 
c_out=1
n_hours,seq_len=24,24
label_len=12
pred_len,out_len=1,1
input_channels,height,width=3,128,128

d_model=256 
n_heads=8
# e_layers=2 
d_layers=2
d_ff=512
dropout=0
learning_rate=0.001
patience=10
image_data_path='2021_9_10_images_path.csv'
text_data_path='2021_9_10_text_data.csv'
resize=128
Text_fusion_weights=0.1
Image_fusion_weights=0.9


# data_path="flo2_nowindspeed.csv"
seed = 42  
Save_file_root_path="InformerStack_Train_test_multi_step_ourdatasets"
results_save_path_root="./"+Save_file_root_path+"/"

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'
os.environ['CUDA_LAUNCH_BLOCKING'] = "0,1"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



 
class EvaluationMetrics:
    def __init__(self):
        # You can add any necessary initializations here
        pass
    def rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    def nmae(self, y_true, y_pred):
        return np.mean(np.abs(y_pred-y_true)/ y_true)
        #return mean_absolute_error(y_true, y_pred)/ (np.max(y_true)-np.min(y_true))
    def calculate_rmse(self,predictions, targets):
         
        if len(predictions) != len(targets):
            raise ValueError("The shape of the predicted and true values do not match.")
        
        # ½«Êý×é×ª»»Îª NumPy Êý×é
        predictions = np.array(predictions)
        targets = np.array(targets)

        # ¼ÆËãÔ¤²âÖµÓëÕæÊµÖµÖ®¼äµÄ²îÖµ
        errors = predictions - targets

        # ¼ÆËã¾ù·½¸ùÎó²î
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

    def FS_RMSE(self, model_y_true, model_y_pred, persistence_y_true, persistence_y_pred):
        model_rmse = self.rmse(model_y_true, model_y_pred)
        persistence_rmse = self.rmse(persistence_y_true, persistence_y_pred)
        return 1 - (model_rmse / persistence_rmse)

    def FS_nRMSE(self, model_y_true, model_y_pred, persistence_y_true, persistence_y_pred):
        model_nrmse = self.nrmse(model_y_true, model_y_pred)
        persistence_nrmse = self.nrmse(persistence_y_true, persistence_y_pred)
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
            mape_step=self.mape(target_step,prediction_step)
            smape_step=self.nmae(target_step,prediction_step)
            R2_step=self.R2(target_step,prediction_step)
             
            rmse_values_list.append(rmse_step)
            nrmse_values_list.append(nrmse_step)
            mape_values_list.append(mape_step)
            smape_values_list.append(smape_step)
            R2_values_list.append(R2_step)
            
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
    
    # Éú³ÉÁÐÃû£¨¼ÙÉèÃ¿¸ö²½³¤¶¼ÓÐ10ÁÐ£©
     
    column_names = [f"true_step{i+1}" for i in range(step_count)] + [f"pred_step{i+1}" for i in range(step_count)]
    
    # ´´½¨DataFrame£¬²¢Ê¹ÓÃÁÐÃû
    df = pd.DataFrame(combined_data, columns=column_names)
    
    # CSVÎÄ¼þÃû
    csv_file = save_path_results_name+".csv"
    
    # ½«DataFrameÐ´ÈëCSVÎÄ¼þ
    df.to_csv(csv_file, index=True)
def write_to_csv(single_value, my_list, csv_file):
    # ½«ÁÐ±í×ª»»ÎªDataFrame£¬²¢½«Ã¿¸öÔªËØ×÷ÎªÒ»ÐÐ
    df_list = pd.DataFrame({'M_step_rmse': my_list})

    # ´´½¨°üº¬µ¥¸öÖµµÄDataFrame
    df_single = pd.DataFrame({'all_samples_flattening_rmse': [single_value]})

    # ½«µ¥¸öÖµºÍÁÐ±íµÄDataFrame½øÐÐºÏ²¢£¬²¢½«ÆäÐ´ÈëCSVÎÄ¼þ
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
            writer.writerow(["RMSE", "nRMSE", "MAPE", "sMAPE", "R2"])  # Ð´Èë±êÌâÐÐ
            for vals in zip(rmse, nrmse, mape, smape, r2):
                writer.writerow(vals)  # ÖðÁÐÐ´ÈëÊý¾Ý
        writer.writerow(["", "", "", "", "", "Multi-step prediction of the mean value of each indicator"])  # Ð´ÈëÒ»¸ö¿ÕÐÐºÍ±êÌâÐÐ
        writer.writerow(["", "", "", "", "", ""])  # Ð´Èë¿ÕÐÐ
        writer.writerow(["", "", "", "", "", ""]) 
        writer.writerow(["all_samples_rmse", "nrmse_mean", "mape_mean", "sampe_mean", "R2_mean"]) 
        writer.writerow(extra_data)  # Ð´Èë×·¼ÓµÄÊý¾Ý
        
    print("Êý¾ÝÒÑÐ´Èë", filename)



# µ÷ÓÃº¯Êý

 

df = pd.read_csv('Final_Evaluation_Pre_Real_values.csv',header=0,index_col=0)
#print()
predicted_values = df.iloc[:, -pred_len:] 
#print(predicted_values)
actual_values = df.iloc[:, :pred_len] 
predicted_values=np.array(predicted_values).reshape(-1,pred_len)
actual_values=np.array(actual_values).reshape(-1,pred_len)
preds,trues=predicted_values,actual_values
# rea_csi,theory_ghi,real_ghi=get_test_real_ghi_theroy_clean_ghi_csi(csi_preds,n_hours,pred_len,n_features)  
# pred_ghi=csi_preds*theory_ghi

targets=trues
predictions=preds
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

def Save_data_to_a_specified_file(results_list,extra_data_mean):
      
    # plot_train_test_loss_and_save(train_loss_list,vali_loss_list,test_loss_list,final_results_save_path)
    # write_to_csv(all_samples_flattening_rmse, results_list[0],results_list[1],results_list[2],results_list[3],results_list[4],final_results_save_path+"/Final_Evaluation_Index_Statistics_Results.csv",)
    write_metrics_to_csv("./Final_Evaluation_Index_Statistics_Results_mape.csv", results_list[0],results_list[1],results_list[2],results_list[3],results_list[4], extra_data_mean)
    
    
    # save_trues_pred(targets,predictions,step_count=len(rmse_values_list),save_path_results_name=final_results_save_path+"/Final_Evaluation_Pre_Real_values")
   
    # torch.save(model,final_results_save_path+"/best_model_"+str(all_samples_flattening_rmse )+".pth")
    print("\033[1;32mCongratulations! The model file has been deleted. You are loading the dataset for training\033[0m")   
    # Save_current_script_to_specified_directory(final_results_save_path)



 

# Dataframe_Final_Evaluation_Index = pd.DataFrame(data={'All_samples_flattening_rmse': [all_samples_flattening_rmse],'Step_rmse':np.array(rmse_values_list).reshape(-1, 1)})
# Dataframe_Final_Evaluation_Index = pd.DataFrame(data={'Step_rmse':np.array(rmse_values_list).reshape(-1, 1)})
# Dataframe_Final_Evaluation_Index.to_csv("Final_Evaluation_Index_Statistics_Results.csv", index=None)

# Dataframe_Final_Evaluation_Pre_Real_values=pd.DataFrame(data={'measured value':np.array(targets),'predicted value':np.array(predictions)})
# Dataframe_Final_Evaluation_Pre_Real_values.to_csv("Final_Evaluation_Pre_Real_values.csv",index=None)



Save_data_to_a_specified_file(results_list,extra_data_mean)


# Save_data_to_a_specified_file(final_results_save_path,Dataframe_Final_Evaluation_Index,Dataframe_Final_Evaluation_Pre_Real_values,model,plt)
# retain_min_folder(results_save_path_root)
 
# end = time.time()
# time=format_time(end-start)


torch.cuda.empty_cache()






