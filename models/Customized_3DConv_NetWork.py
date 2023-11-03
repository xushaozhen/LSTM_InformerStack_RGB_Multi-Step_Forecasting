# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:30:29 2023

@author: Liu_Jun_Desktop
"""
import torch
import torch.nn as nn

class Customized_3DConv(nn.Module):
    def __init__(self, input_channels,depth, height, width,label_len,pred_len):
        super(Customized_3DConv, self).__init__()
        self.depth=depth
        self.input_channels=input_channels
        self.height, self.width=height,width
        self.label_len,self.pred_len=label_len,pred_len
        # First ConvLSTM3D
        self.conv1 = nn.Conv3d( self.input_channels, 256, kernel_size=(3, 3, 3), padding=1)
        self.relu1 = nn.ReLU()

        # Second Conv3D
        self.conv2 = nn.Conv3d(256, 128, kernel_size=(3, 3, 3), padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), padding=1)
        self.dropout1 = nn.Dropout(0.1)

        # Third Conv3D
        self.conv3 = nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=1)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), padding=1)
        self.dropout2 = nn.Dropout(0.1)

        # Flattening the tensor for the Dense layers
        self.flatten = nn.Flatten()
        
        # Compute the flattened dimension
        dim = self._get_flatten_dim()
        # print("dim",dim)
        # Dense Layer
        self.dense1 = nn.Linear(dim, 128)
        
        self.sigmoid = nn.Sigmoid()
        self.dropout3 = nn.Dropout(0.1)
        
        # Output Layer
        self.dense2 = nn.Linear(128, self.label_len+self.pred_len)

    def forward(self, x):
        # Transpose the depth and channel dimensions
        x = x.permute(0, 2, 1, 3, 4)
        # print("self.permute\n",x.shape)
        x = self.conv1(x)
        # print("self.conv1\n",x.shape)
        x = self.relu1(x)

        x = self.conv2(x)
        # print("self.conv2\n",x.shape)
        x = self.relu2(x)
        x = self.pool1(x)
        # print("self.pool1\n",x.shape)
        x = self.dropout1(x)

        x = self.conv3(x)
        # print("self.conv3\n",x.shape)
        x = self.relu3(x)
        x = self.pool2(x)
        # print("self.pool2\n",x.shape)
        x = self.dropout2(x)
        # print("self.dropout2\n",x.shape)
        x = self.flatten(x)
        # print("self.flatten\n",x.shape[1])
        x = self.dense1(x)
        # print("self.dense1\n",x.shape)
        x = self.sigmoid(x)
        x = self.dropout3(x)
        # print("self.dropout3\n",x.shape)
        x = self.dense2(x)
        # print("self.dense2\n",x.shape)
        x = self.sigmoid(x)
        # print("self.sigmoid\n",x.shape)
        x = x.unsqueeze(2) 
        # print("self.unsqueeze\n",x.shape)
        return x

    def _get_flatten_dim(self):
        """
        Compute the flatten dimension after Conv3D and MaxPool3D
        """
        x = torch.randn(1, self.depth, self.input_channels,  self.height,  self.width)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        return x.numel()

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    input = torch.rand(32, 8, 3, 32, 32)
    input.float().to(device)
    model=Customized_3DConv(3,8, 32, 32,4,4)
    out=model(input)
