# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 15:00:21 2023

@author: Liu_Jun_Desktop
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 12:55:26 2023

@author: Liu_Jun_Desktop
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 02:52:24 2023

@author: Liu_Jun_Desktop
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 00:39:10 2023

@author: Liu_Jun_Desktop
"""
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
import matplotlib.pyplot as pltmetric
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error #??????
from sklearn.metrics import r2_score#R square
import shutil
import random    
# from models.model import Informer, InformerStack
from models.model_lstm_concat_add_features_weights_add_Customized_3DConv import Informer, InformerStack
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
from sklearn.metrics import mean_absolute_error #\u5e73\u65b9\u7edd\u5bf9\u8bef\u5dee
from sklearn.metrics import r2_score#R square
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
use_amp = False
output_attention = False
inverse = True

n_features,enc_in,dec_in=6,6,6
 
c_out=1
n_hours,seq_len=24,24
label_len=12
pred_len,out_len=6,6
input_channels,height,width=3,32,32

d_model=256
n_heads=8
# e_layers=2 
d_layers=2
d_ff=2048
dropout=0 
learning_rate=0.001
patience=7
image_data_path='2021_9_10_images_path.csv'
text_data_path='2021_9_10_text_data.csv'
resize=32
Text_fusion_weights=0.1
Image_fusion_weights=0.9


# data_path="flo2_nowindspeed.csv"
seed = 42  
Save_file_root_path="InformerStack_Train_test_multi_step_ourdatasets"
results_save_path_root="./"+Save_file_root_path+"/"

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'
os.environ['CUDA_LAUNCH_BLOCKING'] = "0,1"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

model =InformerStack(enc_in,dec_in,c_out,seq_len,label_len, out_len,input_channels,height,width,
            factor=5, d_model=d_model, n_heads=n_heads, e_layers=[3,2,1], d_layers=d_layers, d_ff=d_ff, 
            dropout=0.0, attn='prob', embed='fixed', freq=freq, activation='gelu',
            output_attention = False, distil=True, mix=True,Text_fusion_weights=Text_fusion_weights
            ,Image_fusion_weights=Image_fusion_weights,device=device)
model = model.to(device)

warnings.filterwarnings('ignore')

# class InformerStack(nn.Module):
#     def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
#                 factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
#                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
#                 output_attention = False, distil=True, mix=True,
#                 device=torch.device('cuda:0')):
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
def get_test_real_ghi_theroy_clean_ghi_csi(pre_values,n_hours,pred_len,n_features):
    data = pd.read_csv(text_data_path,header=0,index_col=0)
    origin_values = data.values[:,:]
    origin_values = origin_values.astype('float32')
     
    train_length=int(origin_values.shape[0]*0.7)
    test_length=int(origin_values.shape[0]*0.2)
    val_length=origin_values.shape[0]-train_length-test_length
    
    train_values = origin_values[:train_length,:]
    val_values=origin_values[train_length-n_hours:train_length+val_length,:]
    test_values = origin_values[origin_values.shape[0]-test_length-n_hours:origin_values.shape[0],:]
    
    reframed1 = series_to_supervised(train_values, n_hours, pred_len)
    reframed2 = series_to_supervised(test_values, n_hours, pred_len)
    
    List_of_load_csi=[i for i in range(-n_features*(pred_len-1)-1, 0, n_features)]
    List_of_load_theory_ghi=[i-1 for i in range(-n_features*(pred_len-1)-1, 0, n_features)]
    List_of_load_actal_ghi=[i-2 for i in range(-n_features*(pred_len-1)-1, 0, n_features)]
    # List_of_load_set = [item for item in range(-pred_len, 0)]
    
    real_csi=reframed2.iloc[:pre_values.shape[0],List_of_load_csi]
    Theoretical_clear_sky_factor=reframed2.iloc[:pre_values.shape[0],List_of_load_theory_ghi]
    real_ghi=reframed2.iloc[:pre_values.shape[0],List_of_load_actal_ghi]
    
    transfer_real_ghi_test_real_csi=np.array(real_csi).reshape(-1,pred_len)
    transfer_Theoretical_clear_sky_factor=np.array(Theoretical_clear_sky_factor).reshape(-1,pred_len)
    transfer_real_ghi=np.array(real_ghi).reshape(-1,pred_len)
    
    return transfer_real_ghi_test_real_csi,transfer_Theoretical_clear_sky_factor,transfer_real_ghi

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




class sky_img_rgb(Dataset):
    def __init__(self,resize,root_path, flag='train', image_data_path=image_data_path,n_hours=6):
        super(sky_img_rgb,self).__init__()

        self.resize = resize
        self.root_path=root_path
        self.flag=flag
        self.image_data_path=image_data_path
        self.n_hours=n_hours
        self.df_images_path=self.__read_data__()
        
        self.images_list = self.load_image(self.df_images_path)
        
    def __read_data__(self):
         
        df_images_path = pd.read_csv(os.path.join(self.root_path,
                                          self.image_data_path),header=0,index_col=0)
        train_length=int(df_images_path.shape[0]*0.7)
        test_length=int(df_images_path.shape[0]*0.2)
        val_length=df_images_path.shape[0]-train_length-test_length
        
        train_values = df_images_path.iloc[:train_length,:]
        val_values=df_images_path.iloc[train_length-self.n_hours:train_length+val_length,:]
        test_values = df_images_path.iloc[df_images_path.shape[0]-test_length-self.n_hours:df_images_path.shape[0],:]
        if self.flag == 'train':#80%
            df_images_path = series_to_supervised(train_values, self.n_hours, pred_len).values[:,:-pred_len]
        elif self.flag == 'val':
            df_images_path =series_to_supervised(val_values, self.n_hours, pred_len).values[:,:-pred_len]
        else:
            df_images_path =series_to_supervised(test_values, self.n_hours, pred_len).values[:,:-pred_len]
       
        return df_images_path
    def load_image(self,img_path):
        images_list = []
        # print(len(img_path))
        for row in img_path:
            images_list.append(row)
        print(len(images_list))
        return images_list

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        imgs = self.images_list[index]
        # print(imgs)
        total_img = []
        for i,x in enumerate(imgs):
            img = np.array(Image.open(x).convert('RGB'))
            # print(img.shape)
            transform_list = [transforms.ToTensor(),transforms.Resize((self.resize,self.resize)),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
            transformer = transforms.Compose(transform_list)
            tran_img = transformer(img)
            # print(tran_img.shape)
            tran_img = torch.unsqueeze(tran_img,0)
            # print(tran_img.shape)
            if i == 0:
                multi_tran_img = torch.cat((tran_img,tran_img),dim=0)
            elif i > 0:
                multi_tran_img = torch.cat((multi_tran_img,tran_img),dim=0)
        return multi_tran_img[1:,:,:,:]

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='MS', text_data_path=text_data_path, 
                 target='ghi', scale=True, inverse=False, timeenc=0, freq=freq, cols=None,Dataset_partition_ratio=[0.7,0.2]):
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
        self.text_data_path = text_data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.text_data_path))
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
        if self.set_type==0 or self.set_type==1:
        
            seq_x = self.add_random_noise(seq_x)
        else:
            seq_x=seq_x
         
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
    def add_random_noise(self, sequence, noise_level=0.1):
        # Èç¹ûÊÇÔÚCUDAµÄtensor£¬½«ÆäÒÆµ½CPUÉÏ²¢×ª»»ÎªnumpyÊý×é
        if isinstance(sequence, torch.Tensor) and sequence.is_cuda:
            sequence = sequence.cpu().numpy()
        
        # Èç¹û²»ÊÇnumpyÊý×é£¬³¢ÊÔ×ª»»
        if not isinstance(sequence, np.ndarray):
            sequence = np.array(sequence)
            
        noise = np.random.normal(loc=0, scale=noise_level, size=sequence.shape)
        noisy_sequence = sequence + noise
        
        return torch.tensor(noisy_sequence)
    
    
class Combined_image_text_Dataset(Dataset):
   def __init__(self,flag):
       self.flag=flag
       self.image_dataset = sky_img_rgb(resize=resize,root_path='./', flag=self.flag, image_data_path=image_data_path,n_hours=n_hours)
       self.text_dataset = Dataset_Custom('./', flag=self.flag, size=[seq_len,label_len,pred_len], features='MS', 
                                          text_data_path=text_data_path, target='ghi', scale=True, inverse=True, timeenc=0,
                                          freq=freq, cols=None,Dataset_partition_ratio=[0.7,0.2])

   def __getitem__(self, index):
       seq_x, seq_y, seq_x_mark, seq_y_mark = self.text_dataset[index]
       images = self.image_dataset[index]
       
       return seq_x,seq_y,seq_x_mark,seq_y_mark,images

   def __len__(self):
       # È·±£Á½¸öÊý¾Ý¼¯µÄ³¤¶ÈÏàÍ¬£¬»òÕßÄú¿ÉÒÔ¸ù¾ÝÐèÒªÐÞ¸Ä´ËÂß¼­
       # assert len(self.image_dataset) == len(self.text_dataset)
       return len(self.image_dataset)
    
def _process_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark,images,features="MS"):
       batch_x = batch_x.float().to(device)
       batch_y = batch_y.float().to(device)

       batch_x_mark = batch_x_mark.float().to(device)
       batch_y_mark = batch_y_mark.float().to(device)
       images=images.float().to(device)
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
                   outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark,images)[0]
               else:
                   outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark,images)
       else:
           if output_attention:
               outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark,images)[0]
           else:
               outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark,images)
               # print("liujun:",outputs.shape)
               # print("liujun:",outputs[0].shape)
               # print("liujun:",outputs[0])
       if inverse:
           outputs = dataset_object.text_dataset.inverse_transform(outputs)
       f_dim = -1 if features=='MS' else 0
       batch_y = batch_y[:,-pred_len:,f_dim:].to(device)

       return outputs, batch_y

# batch_size=800
#test

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
def Save_current_script_to_specified_directory(final_results_save_path):  #½«Ã¿¸öµ¥´ÊÓÃÏÂ»®ÏßÁ´½ÓÊä³ö
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
        
def Save_data_to_a_specified_file(final_results_save_path,Dataframe_Final_Evaluation_Index,Dataframe_Final_Evaluation_Pre_Real_values,model,plt):
      
    plot_train_test_loss_and_save(train_loss_list, vali_loss_list,test_loss_list, final_results_save_path)
    # df2 = pd.DataFrame(data={'RMSE': [rmse1], 'nRMSE': [nrmse1], 'MAE': [mae1], 'R^2': [R21]})
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
def _get_data(root_path="./",flag="test",size=[seq_len,label_len,pred_len],
              features="MS",target="ghi",inverse=True,timeenc = 0,
              freq=freq,cols=None,batch_size=batch_size,num_workers=0):
        
        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = batch_size; freq=freq
        # elif flag=='pred':
        #     shuffle_flag = False; drop_last = False; batch_size = 1; freq=detail_freq
        #     Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = False; batch_size = batch_size; freq=freq
        text_image_dataset =Combined_image_text_Dataset(flag)
        
        print(flag, len(text_image_dataset))
        text_image_data_loader = DataLoader(
            text_image_dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=num_workers,
            drop_last=drop_last)
       

        return text_image_dataset, text_image_data_loader

def adjust_learning_rate(optimizer, epoch, lradj='type1',learning_rate=learning_rate):
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
       for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,images) in enumerate(vali_loader):
           pred, true = _process_one_batch(
               vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark,images,features="MS")
           loss = criterion(pred.detach().cpu(), true.detach().cpu())
           total_loss_list.append(loss)
       total_loss = np.average(total_loss_list)
       model.train()
       return total_loss_list,total_loss
    
def train(use_amp=True):
        train_text_image_dataset, train_text_image_data_loader =_get_data(flag = 'train')
        vali_text_image_dataset,vali_text_image_data_loader = _get_data(flag = 'val')
        test_text_image_dataset,test_text_image_data_loader= _get_data(flag = 'test')
       
        early_stopping = EarlyStopping()
        # path = os.path.join(checkpoints, se torch.cuda.empty_cache()tting)
        # if not os.path.exists(path):
        #     os.makedirs(path)

        # time_now = time.time()
        
        train_steps = len(train_text_image_data_loader)
        early_stopping = EarlyStopping(patience=7, verbose=True)
        
        model_optim = _select_optimizer(model,learning_rate=learning_rate)
        criterion =  _select_criterion()
        train_loss_list1 =[] 
        vali_loss_list1=[]
        test_loss_list1=[]
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(train_epochs):
            iter_count = 0
            train_loss_list=[]
            
            model.train()
            # epoch_time = time.time()
            for (batch_x,batch_y,batch_x_mark,batch_y_mark,images) in tqdm(train_text_image_data_loader,bar_format="{l_bar}\033[31m{bar}\033[0m{r_bar}"):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = _process_one_batch(
                    train_text_image_dataset, batch_x, batch_y, batch_x_mark, batch_y_mark,images)
                loss = criterion(pred, true)
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
            vali_loss_list,vali_loss = vali(vali_text_image_dataset,vali_text_image_data_loader, criterion)
            test_loss_list,test_loss = vali(test_text_image_dataset,test_text_image_data_loader, criterion)
            
            train_loss_list1.append(train_loss)
            vali_loss_list1.append(vali_loss)
            test_loss_list1.append(test_loss)
            print("Epoch: {0} \n Train Loss: {1:.7f} \n Vali Loss: {2:.7f} \n Test Loss: {3:.7f}".format(
             epoch + 1, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, model, ".")
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch, lradj='type1',learning_rate=learning_rate)
            
        best_model_path = './'+'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))
        
        return model,train_loss_list1,vali_loss_list1,test_loss_list1
   
def test():
        test_text_image_dataset,test_text_image_data_loader=_get_data(flag='test')
        
        model.eval()
        
        preds = []
        trues = []
        
        for (batch_x,batch_y,batch_x_mark,batch_y_mark,images) in tqdm(test_text_image_data_loader,bar_format="{l_bar}\033[31m{bar}\033[0m{r_bar}"):
            pred, true = _process_one_batch(
                test_text_image_dataset, batch_x, batch_y, batch_x_mark, batch_y_mark,images,features="MS")
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
        return preds,trues


def plot_train_test_loss_and_save(train_loss_list,vali_loss_list,test_loss_list,save_path):
    plt.figure()
    plt.plot(train_loss_list, label='Training loss')
    plt.plot(vali_loss_list, label='Vali loss')
    plt.plot(test_loss_list, label='Test loss')
    plt.xlabel('Epoch',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.legend(fontsize=14)
    plt.autoscale()
    plt.savefig(save_path+"/loss_plot.png")
    plt.show()
class EvaluationMetrics:
    def __init__(self):
        # You can add any necessary initializations here
        pass
    def rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
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
            mape_step=self.medape(target_step,prediction_step)
            smape_step=self.smape(target_step,prediction_step)
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

 



model,train_loss_list,vali_loss_list,test_loss_list=train(use_amp=False)
preds,trues=test()
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

def Save_data_to_a_specified_file(train_loss_list,vali_loss_list,test_loss_list,
                                  final_results_save_path,rmse_values_list,model,all_samples_flattening_rmse):
      
    plot_train_test_loss_and_save(train_loss_list,vali_loss_list,test_loss_list,final_results_save_path)
    # write_to_csv(all_samples_flattening_rmse, results_list[0],results_list[1],results_list[2],results_list[3],results_list[4],final_results_save_path+"/Final_Evaluation_Index_Statistics_Results.csv",)
    write_metrics_to_csv(final_results_save_path+"/Final_Evaluation_Index_Statistics_Results.csv", results_list[0],results_list[1],results_list[2],results_list[3],results_list[4], extra_data_mean)
    
    
    save_trues_pred(targets,predictions,step_count=len(rmse_values_list),save_path_results_name=final_results_save_path+"/Final_Evaluation_Pre_Real_values")
   
    torch.save(model,final_results_save_path+"/best_model_"+str(all_samples_flattening_rmse )+".pth")
    print("\033[1;32mCongratulations! The model file has been deleted. You are loading the dataset for training\033[0m")   
    Save_current_script_to_specified_directory(final_results_save_path)

def plot_multi_steps_rmse(multi_steps_rmse_list,save_path):
    plt.figure()
    plt.plot(multi_steps_rmse_list,color='r')
    
    plt.xlabel('step',fontsize=16)
    plt.ylabel('rmse',fontsize=16)
    plt.yticks( fontsize=16)
    plt.title('Multi-step prediction of RMSE curves',fontsize=16)
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
    plt.savefig(save_path+"/Multi-step prediction of RMSE curves.png",dpi=600)
    plt.show()

 

# Dataframe_Final_Evaluation_Index = pd.DataFrame(data={'All_samples_flattening_rmse': [all_samples_flattening_rmse],'Step_rmse':np.array(rmse_values_list).reshape(-1, 1)})
# Dataframe_Final_Evaluation_Index = pd.DataFrame(data={'Step_rmse':np.array(rmse_values_list).reshape(-1, 1)})
# Dataframe_Final_Evaluation_Index.to_csv("Final_Evaluation_Index_Statistics_Results.csv", index=None)

# Dataframe_Final_Evaluation_Pre_Real_values=pd.DataFrame(data={'measured value':np.array(targets),'predicted value':np.array(predictions)})
# Dataframe_Final_Evaluation_Pre_Real_values.to_csv("Final_Evaluation_Pre_Real_values.csv",index=None)


model_files_automatically_deleted(final_results_save_path)
Save_data_to_a_specified_file(train_loss_list,vali_loss_list,test_loss_list,
                                  final_results_save_path,rmse_values_list,model,all_samples_flattening_rmse)
plot_multi_steps_rmse(rmse_values_list,final_results_save_path)

# Save_data_to_a_specified_file(final_results_save_path,Dataframe_Final_Evaluation_Index,Dataframe_Final_Evaluation_Pre_Real_values,model,plt)
# retain_min_folder(results_save_path_root)
 
end = time.time()
time=format_time(end-start)

print("\033[1;32m Total code running time:\033[0m",time)
torch.cuda.empty_cache()





