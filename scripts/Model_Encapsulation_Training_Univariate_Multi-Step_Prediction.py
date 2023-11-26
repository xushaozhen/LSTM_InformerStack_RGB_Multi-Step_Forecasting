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
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as pltmetric
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score
import shutil
import random
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.timefeatures import time_features
from utils.tools import StandardScaler
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
from sklearn.metrics import r2_score#R square
import os
import time
import matplotlib.pyplot as plt
import warnings
import re



n_hours = 24
n_features = 4
train_epochs = 100
batch_size=100

padding=0
use_amp = False
output_attention = False
inverse = True


seq_len=24 
label_len=12
out_len=24
pred_len=24
d_model=4
n_heads=4
e_layers=3 
d_layers=2 
d_ff=512
dropout=0

patience=10
data_path="flo2_nowindspeed.csv"
seed = 42  
Save_file_root_path="JunLiu_Informer_Model_Encapsulation_Training_Univariate_Multi-Step_Prediction"
results_save_path_root="./"+Save_file_root_path+"/"

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

os.environ['CUDA_LAUNCH_BLOCKING'] = "0,1"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model =Informer(enc_in=4, dec_in=4, c_out=1, seq_len=seq_len, label_len=label_len, out_len=out_len, 
            factor=5, d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_layers=d_layers, d_ff=512, 
            dropout=dropout, attn='prob', embed='fixed', freq='h', activation='gelu', 
            output_attention = False, distil=True, mix=True,
            device=device)
model = model.to(device)

warnings.filterwarnings('ignore')

class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab,d_model)
        self.d_model = d_model
    def forward(self,x):
        return self.lut(x) * math.sqrt(self.d_model)

#?????????
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()


# print(model)
# model = model.to(torch.device('cuda:0'))  # 如果有可用的GPU，将模型放到GPU上


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

 


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='MS', data_path='flo2_nowindspeed.csv', 
                 target='ghi', scale=True, inverse=False, timeenc=0, freq='h', cols=None,Dataset_partition_ratio=[0.7,0.2]):
        # size [seq_len, label_len, pred_len]
        # info
       
        if size == None:
            self.seq_len = 24
            self.label_len = 12
            self.pred_len =12
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        self.Dataset_partition_ratio=Dataset_partition_ratio
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        num_train = int(len(df_raw)*self.Dataset_partition_ratio[0])
        num_test = int(len(df_raw)*self.Dataset_partition_ratio[1])
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
       
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
def _process_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark,features="MS"):
       batch_x = batch_x.float().to(device)
       batch_y = batch_y.float().to(device)

       batch_x_mark = batch_x_mark.float().to(device)
       batch_y_mark = batch_y_mark.float().to(device)

       # decoder input
       if padding==0:
           # dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
           dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float().to(device)
       elif padding==1:
           dec_inp = torch.ones([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()
       # dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
       dec_inp = torch.cat([batch_y[:,:label_len,:], dec_inp], dim=1).float().to(device)
       # encoder - decoder
       if use_amp:
           with torch.cuda.amp.autocast():
               if  output_attention:
                   outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
               else:
                   outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
       else:
           if output_attention:
               outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
           else:
               outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
       if inverse:
           outputs = dataset_object.inverse_transform(outputs)
       f_dim = -1 if features=='MS' else 0
       batch_y = batch_y[:,-pred_len:,f_dim:].to(device)

       return outputs, batch_y

# batch_size=800
#test

class EarlyStopping:
    def __init__(self, patience=patience, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss
        

def retain_min_folder(results_save_path_root):
   
    folder_list = os.listdir(results_save_path_root)

    pattern = r'\d+\.\d+'
    folder_numbers = [float(re.search(pattern, folder_name).group()) for folder_name in folder_list if re.search(pattern, folder_name)]

    if len(folder_numbers)>=2:
 
        min_folder_number = min(folder_numbers)
        min_folder_name = Save_file_root_path+"_"+str(min_folder_number)

        for folder_name in folder_list:

            if folder_name.startswith(Save_file_root_path+"_") and folder_name != min_folder_name:
                wait_del_file_path = os.path.join(results_save_path_root, folder_name)
                shutil.rmtree(wait_del_file_path)
                print(f'Deleted folder: {wait_del_file_path}')

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
    current_file_path = os.path.abspath(__file__)
    current_file_name = os.path.basename(__file__)
    destination_folder = final_results_save_path
    
    destination_file_path = os.path.join(destination_folder, current_file_name)
    
  
    shutil.copy(current_file_path, destination_file_path)
        
def Save_data_to_a_specified_file(final_results_save_path,Dataframe_Final_Evaluation_Index,Dataframe_Final_Evaluation_Pre_Real_values,model,plt):
      
    plot_train_test_loss_and_save(train_loss_list, vali_loss_list,test_loss_list, final_results_save_path)
    # df2 = pd.DataFrame(data={'RMSE': [rmse1], 'nRMSE': [nrmse1], 'SMAPE': [mae1], 'R^2': [R21]})
    Dataframe_Final_Evaluation_Index.to_csv(final_results_save_path+"/Final_Evaluation_Index_Statistics_Results.csv", index=None)
    # df3=pd.DataFrame(data={'measured value':np.array(test_real_value).reshape(-1,),'predicted value':np.array(final_pre_value).reshape(-1,)})
    Dataframe_Final_Evaluation_Pre_Real_values.to_csv(final_results_save_path+"/Final_Evaluation_Pre_Real_values.csv",index=None)
    torch.save(model,final_results_save_path+"/best_model_"+str(all_samples_flattening_rmse)+".pth")
    print("\033[1;32mCongratulations! The model file has been deleted. You are loading the dataset for training\033[0m")   
    Save_current_script_to_specified_directory(final_results_save_path)

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    time="%02d:%02d:%02d" % (h, m, s)
    return time
def _get_data(root_path="./",data_path=data_path,
              flag="test",size=[seq_len,label_len,pred_len],
              features="MS",target="ghi",inverse=True,timeenc = 0,
              freq='h',cols=None,batch_size=batch_size,num_workers=0):
        
        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = batch_size; freq=freq
        # elif flag=='pred':
        #     shuffle_flag = False; drop_last = False; batch_size = 1; freq=detail_freq
        #     Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = batch_size; freq=freq
        data_set = Dataset_Custom(
            root_path=root_path,
            data_path=data_path,
            flag=flag,
            size=size,
            features=features,
            target=target,
            inverse=inverse,
            timeenc=timeenc,
            freq=freq,
            cols=cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=num_workers,
            drop_last=drop_last)

        return data_set, data_loader

def adjust_learning_rate(optimizer, epoch, lradj='type1',learning_rate=0.001):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if lradj=='type1':
        lr_adjust = {epoch:learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
        
def _select_optimizer(model,learning_rate):
       model_optim = optim.Adam(model.parameters(), lr=learning_rate)
       return model_optim
    
def _select_criterion():
    criterion =  nn.MSELoss()
    return criterion
    
def vali(vali_data, vali_loader, criterion):
       model.eval()
       total_loss_list = []
       for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
           pred, true = _process_one_batch(
               vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark,features="MS")
           loss = criterion(pred.detach().cpu(), true.detach().cpu())
           total_loss_list.append(loss)
       total_loss = np.average(total_loss_list)
       model.train()
       return total_loss_list,total_loss
    
def train(use_amp=True):
        train_data, train_loader =_get_data(flag = 'train')
        vali_data, vali_loader = _get_data(flag = 'val')
        test_data, test_loader = _get_data(flag = 'test')
        early_stopping = EarlyStopping()
        # path = os.path.join(checkpoints, setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)

        # time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        model_optim = _select_optimizer(model,learning_rate=0.001)
        criterion =  _select_criterion()

        if use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(train_epochs):
            iter_count = 0
            train_loss_list = []
            
            model.train()
            # epoch_time = time.time()
            for (batch_x,batch_y,batch_x_mark,batch_y_mark) in tqdm(train_loader,bar_format="{l_bar}\033[31m{bar}\033[0m{r_bar}"):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = _process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark,features="MS")
                loss = criterion(pred, true)
                print("losssssssss.shape:",loss.shape)
                train_loss_list.append(loss.item())
                
                # if (i+1) % 100==0:
                #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                #     speed = (time.time()-time_now)/iter_count
                #     left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()
                
                if use_amp:
                    scaler.scale(loss.half()).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {}".format(epoch+1))
            train_loss = np.average(train_loss_list)
            vali_loss_list,vali_loss = vali(vali_data, vali_loader, criterion)
            test_loss_list,test_loss = vali(test_data, test_loader, criterion)

            print("Epoch: {0} \n Train Loss: {1:.7f} \n Vali Loss: {2:.7f} \n Test Loss: {3:.7f}".format(
             epoch + 1, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, model, ".")
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch, lradj='type1',learning_rate=0.001)
            
        best_model_path = './'+'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))
        
        return model,train_loss_list,vali_loss_list,test_loss_list
   
def test():
        test_data, test_loader =_get_data(flag='test')
        
        model.eval()
        
        preds = []
        trues = []
        
        for (batch_x,batch_y,batch_x_mark,batch_y_mark) in tqdm(test_loader,bar_format="{l_bar}\033[31m{bar}\033[0m{r_bar}"):
            pred, true = _process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark,features="MS")
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape_before:', preds.shape, trues.shape)
        # preds = preds.reshape(preds.shape[0]*preds.shape[1],1)
        # trues = trues.reshape(trues.shape[0]*trues.shape[1],1)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape_after:', preds.shape, trues.shape)
        preds = np.transpose(preds, (0, 2, 1))
        trues = np.transpose(trues, (0, 2, 1))
        print('test shape_after_transpose:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-1])
        print('test shape_after_2:', preds.shape, trues.shape)
        # result save
        # folder_path = './results/' 
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        # print('mse:{}, mae:{}'.format(mse, mae))

        # np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path+'pred.npy', preds)
        # np.save(folder_path+'true.npy', trues)

        return preds,trues

# def predict(self, setting, load=False):
#     pred_data, pred_loader = self._get_data(flag='pred')
    
#     if load:
#         path = os.path.join(self.args.checkpoints, setting)
#         best_model_path = path+'/'+'checkpoint.pth'
#         self.model.load_state_dict(torch.load(best_model_path))

#     self.model.eval()
    
#     preds = []
    
#     for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
#         pred, true = self._process_one_batch(
#             pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
#         preds.append(pred.detach().cpu().numpy())

#     preds = np.array(preds)
#     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    
#     # result save
#     folder_path = './results/' + setting +'/'
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
    
#     np.save(folder_path+'real_prediction.npy', preds)
    
#     return


def plot_train_test_loss_and_save(train_losses_list,vali_loss_list,test_losses_list,save_path):
    plt.plot(train_loss_list, label='Training loss')
    plt.plot(vali_loss_list, label='Vali loss')
    plt.plot(test_loss_list, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path+"/loss_plot.png")
    plt.show()


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





model,train_loss_list,vali_loss_list,test_loss_list=train(use_amp=False)
preds,trues=test()
targets=trues
predictions=preds
all_samples_flattening_rmse=RMSE_after_flattening_of_all_samples(predictions, targets)
rmse_values_list=calculate_multi_step_all_samples_rmse(predictions, targets)


# rmse1 = np.sqrt(mean_squared_error(test_real_value,final_pre_value))
# nrmse1=rmse1/(np.mean(test_real_value,axis=0))
# smape1=mean_absolute_error(test_real_value,final_pre_value)
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
print("\033[1;32m******main_output\u53c2\u6570\u8bc4\u4f30\u7ed3\u679c******\033[0m")



write_to_csv(all_samples_flattening_rmse, rmse_values_list, "Final_Evaluation_Index_Statistics_Results.csv",)
save_trues_pred(trues,preds,step_count=len(rmse_values_list),save_path_results_name="Final_Evaluation_Pre_Real_values")

# Dataframe_Final_Evaluation_Index = pd.DataFrame(data={'All_samples_flattening_rmse': [all_samples_flattening_rmse],'Step_rmse':np.array(rmse_values_list).reshape(-1, 1)})
# Dataframe_Final_Evaluation_Index = pd.DataFrame(data={'Step_rmse':np.array(rmse_values_list).reshape(-1, 1)})
# Dataframe_Final_Evaluation_Index.to_csv("Final_Evaluation_Index_Statistics_Results.csv", index=None)

# Dataframe_Final_Evaluation_Pre_Real_values=pd.DataFrame(data={'measured value':np.array(targets),'predicted value':np.array(predictions)})
# Dataframe_Final_Evaluation_Pre_Real_values.to_csv("Final_Evaluation_Pre_Real_values.csv",index=None)


# model_files_automatically_deleted(final_results_save_path)
# Save_data_to_a_specified_file(final_results_save_path,Dataframe_Final_Evaluation_Index,Dataframe_Final_Evaluation_Pre_Real_values,model,plt)
# retain_min_folder(results_save_path_root)

end = time.time()
time=format_time(end-start)

print("\033[1;32m??????:\033[0m",time)


