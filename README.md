# LSTM-InformerStack_RGB_Multi-Step_Forecasting(LIRMSF)
LIRMSF is an open source library for deep learning researchers, especially for long time series forecasting.

We provide a concise codebase for predicting solar irradiance multi-step time series using deep learning models that combine meteorological text data with all-sky image data. Eight different model prediction methods are covered: peresietence, LSTM-RGB,Transformer,LSTM-Transformer,LSTM-Transformer-RGB,InformerStack,LSTM-InformerStack,LSTM- InformerStack-RGB.**

** If you have proposed advanced and awesome models, welcome to send your paper/code link to us or raise a pull request. We will add them to this repo as soon as possible.


## Usage

1. Install 
If you are working on windows, you need to first install PyTorch with pip install torch -f https://download.pytorch.org/whl/torch_stable.html. Otherwise, you can install the package via conda.
You also install pip install Python 3.8. For convenience, execute the following command.
```
pip install -r requirements.txt

```
2. Prepare Data. You can obtained the datasets from [[Google Drive]](https://drive.google.com/drive/folders/1ASQF064ZEAAqWNIUc1-F0vPPQ01eMxKo?usp=sharing).Then place the downloaded data under the folder `./dataset`. 

3. Train and evaluate model. We provide available all benchmark experiment scripts under the folder `./scripts/` with training scripts for the model. In addition, we performed predictions with different time steps (1, 6, 12, and 24) for different prediction methods and evaluated the models according to different evaluation metrics.You can reproduce the experiment results as the following examples:

### The basic parameter settings for model training.

|Methods|Original data source|Image/text input shape|Major models|Output shape
--------|---------------------|----------------------|------------|-----------
peresietence|/|/|/|/|
LSTM-RGB|RGB images and text|(N,T,C,W,H)/(N,T,F)|LSTM|(N,S)
Transformer|text data|/(N,T,F)|Transformer|(N,S)
LSTM-Transformer|text data|/(N,T,F)|LSTM-Transformer|(N,S)
LSTM-Transformer-RGB|RGB images and text|(N,T,C,,W,H)/(N,T,F)|LSTM-Transformer|(N,S)
InformerStack|text data|/(N,T,F)|InformerStack|(N,S)
LSTM-InformerStack|text data|/(N,T,F)|LSTM-InformerStack|(N,S)
LSTM-InformerStack-RGB|RGB images and text|(N,T,C,,W,H)/(N,T,F)|LSTM-InformerStack|(N,S)

* N,T,C,W,H,F,S represent batchsize,timestep,number of channels,image width and height, number of features, number of prediction steps respectively.*



4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/model_lstm_concat.py`.
- Include the newly added model in the  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.

## Usage example
Networks can be trained on pandas Dataframes where the collected dataset is first converted into a time series dataset.

```
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

```

