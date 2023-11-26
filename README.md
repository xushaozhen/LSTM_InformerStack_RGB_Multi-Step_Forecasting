# LSTM-InformerStack_RGB_Multi-Step_Forecasting(LIRMSF)
LIRMSF is an open source library for deep learning researchers, especially for long time series forecasting.

We provide a concise codebase for predicting solar irradiance multi-step time series using deep learning models that combine meteorological text data with all-sky image data. Eight different model prediction methods are covered: peresietence, LSTM-RGB,Transformer,LSTM-Transformer,LSTM-Transformer-RGB,InformerStack,LSTM-InformerStack,LSTM- InformerStack-RGB.**

** If you have proposed advanced and awesome models, welcome to send your paper/code link to us or raise a pull request. We will add them to this repo as soon as possible.


## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtained the datasets from [[Google Drive]](https://drive.google.com/drive/folders/1ASQF064ZEAAqWNIUc1-F0vPPQ01eMxKo?usp=sharing).Then place the downloaded data under the folder `./dataset`. 

3. Train and evaluate model. We provide available all benchmark experiment scripts under the folder `./scripts/` with training scripts for the model. You can reproduce the experiment results as the following examples:

```
# persistence
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
# short-term forecast
bash ./scripts/short_term_forecast/TimesNet_M4.sh
# imputation
bash ./scripts/imputation/ETT_script/TimesNet_ETTh1.sh
# anomaly detection
bash ./scripts/anomaly_detection/PSM/TimesNet.sh
# classification
bash ./scripts/classification/TimesNet.sh
```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.

## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Learning Representations},
  year={2023},
}
```

## Contact
If you have any questions or suggestions, feel free to contact:

- Haixu Wu (whx20@mails.tsinghua.edu.cn)
- Tengge Hu (htg21@mails.tsinghua.edu.cn)
- Haoran Zhang (z-hr20@mails.tsinghua.edu.cn)

or describe it in Issues.

## Acknowledgement

This library is constructed based on the following repos:

- Forecasting: https://github.com/thuml/Autoformer

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer

- Classification: https://github.com/thuml/Flowformer

All the experiment datasets are public and we obtain them from the following links:

- Long-term Forecasting and Imputation: https://github.com/thuml/Autoformer

- Short-term Forecasting: https://github.com/ServiceNow/N-BEATS

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer

- Classification: https://www.timeseriesclassification.com/
