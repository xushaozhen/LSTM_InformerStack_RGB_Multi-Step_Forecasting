# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:14:00 2023

@author: xsz
"""
import pandas as pd
import numpy as np
df = pd.read_csv('Final_Evaluation_Pre_Real_values.csv')
#print()
predicted_values = df.iloc[:, 2] 
#print(predicted_values)
actual_values = df.iloc[:, 1]

 
#print(actual_values)

def mape(self, actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual))
mape_values=mape(actual, pred)
mape_values_list=[]
mape_values_list.append()
 