# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 03:27:27 2023

@author: Liu_Jun_Desktop
"""
import pandas as pd
import numpy as np

class Data_deletion:
    def __init__(self, input_file: str, n_in:int, n_features:int,output_path:str):
        self.input_file = input_file
        self.n_in = n_in
        self.n_features = n_features
        self.output_path=output_path
        self.df = pd.read_csv(self.input_file, header=0, index_col=0)
        self.save_to_csv(self.output_path)
        
    @staticmethod
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

    @staticmethod
    def extract_date(path: str) -> str:
        date_part = path.split('_')[2]
        return date_part[:8]

    def process_dataframe(self) -> pd.DataFrame:
        dataframe = self.series_to_supervised(self.df, n_in=self.n_in, n_out=1, dropnan=True)

        rows_to_drop = []
        for index, row in dataframe.iterrows():
            dates = [self.extract_date(str(cell)) for cell in row if isinstance(cell, str) and '_CAM1_' in cell]
            if len(set(dates)) > 1:
                rows_to_drop.append(index)

        dataframe.drop(rows_to_drop, inplace=True)

        sub_series_a = dataframe.iloc[:, :self.n_features]
        sub_series_b = np.array(dataframe.iloc[-1, self.n_features:]).reshape(-1, self.n_features)

        result = np.concatenate((sub_series_a, sub_series_b), axis=0)
        cleaning_data = pd.DataFrame(result)
        cleaning_data.columns = ['rgb']

        return cleaning_data

    def save_to_csv(self, output_file: str) -> None:
        cleaned_data = self.process_dataframe()
        cleaned_data.to_csv(output_file, index=True)

# 使用

 

class DataLookupAndMatch:
    def __init__(self,file1_path:str, file2_path:str,output_path:str):
        self.file1_path = file1_path
        self.file2_path = file2_path
        self.output_path=output_path
        self.df1_original = pd.read_csv(self.file1_path, header=0, index_col=0)
        self.df2 = pd.read_csv(self.file2_path)
       
         
        self.process()
        self.check_lengths()
        self.save_output()

    @staticmethod
    def extract_timestamp(path):
        timestamp = path.split('_')[2] + "_" + path.split('_')[3]
        return timestamp

    @staticmethod
    def convert_timestamp(timestamp):
        date, time = timestamp.split('_')
        formatted_date = f"{date[:4]}/{int(date[4:6])}/{int(date[6:8])} {int(time[:2])}:{time[2:4]}"
        return formatted_date

    def process(self):
        df1 = self.df1_original.copy()
        df1['date'] = df1['rgb'].apply(lambda x: self.convert_timestamp(self.extract_timestamp(x)))

        result_df = df1.merge(self.df2, on='date', how='inner')

        self.output = result_df[['date', 'Temp', 'Hum', 'Height', 'csi', 'Theory', 'ghi']]

    def save_output(self):
        if self.output is not None:
            self.output.set_index('date').to_csv(self.output_path)
            print("\033[1;32m Congratulations! Data Cleaning Complete\033[0m")
            
        else:
            print("Please process the data first.")

    def check_lengths(self):
        if self.output is not None:
            if len(self.output) == len(self.df1_original):
                print("The new CSV has the same number of rows as the original CSV.")
                print("\033[1;32m Congratulations! Data Cleaning Complete\033[0m")
            else:
                print(f"The new CSV has {len(self.output)} rows while the original CSV has {len(self.df1_original)} rows.")
        else:
            print("Please process the data first.")

# 使用类处理CSV
if __name__=="__main__":
    processor1 = Data_deletion('original_2021_9_10_images_path.csv', 24,1,"2021_9_10_images_path.csv")
    
    processor2 = DataLookupAndMatch('2021_9_10_images_path.csv','original_wait_find_2021_9_10_text_data.csv','2021_9_10_text_data.csv')
