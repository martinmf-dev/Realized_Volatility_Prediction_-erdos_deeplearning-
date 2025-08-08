# Realized Volatility Prediction 

## Folder and file organization 
<pre>
  .
  ├── data_collecting 
      ├── data_collecting_kaggle.ipynb
      └── data_collecting_kaggle.py 
  ├── data_processing
      ├── baseline_model.ipynb
      ├── create_datasets.ipynb
      ├── data_processing_functions.ipynb 
      ├── first_look.ipynb 
      ├── missing_time_points.ipynb 
      └── understanding_time_id_recovery.ipynb 
  ├── dotenv_env
  ├── EDA
      ├── EDA_martin.ipynb
      └── EDA_yuan.ipynb 
  ├── ETLpipeline
      ├── Default_Extract_Transform.py
      └── Loading_Example.ipynb 
  ├── model_data 
  ├── NNetwork
      ├── Decoder_transformer.ipynb
      ├── Frozen_conv_layer.ipynb 
      ├── Learned_emb_RNN.ipynb
      ├── Optimizing_hyperparameters.ipynb
      ├── Parameter_embedding.ipynb
      ├── RNN_with_fronzen_conv.ipynb 
      └── Transformer_with_frozen_conv_1.ipynb 
  ├── processed_data 
  ├── proj_mod 
      ├── __pycache__
      ├── __init__.py
      ├── data_processing.py 
      ├── recover_time_id.py 
      ├── training.py 
      └── visualization.py 
  └── raw_data 
</pre>

## Folder and file explanations 

* ./data_collecting: Contains instruction and code for downloading raw data from kaggle.
* ./data_processing:
  - basline_model.ipynb: The baseline model.
  - create_datasets.ipynb: Documentation of the pytorch Dataset subclass (with source code in "./proj_mod/training.py").
  - data_processing_functions.ipynb: Documentation of data processing functions (with source code in "./proj_mod/data_processing.py").
  - first_look.ipynb: A first look at the raw data.
  - missing_time_points.ipynb: Documentation on which stocks are missing which time id's.
  - understanding_time_id_recovery.ipynb: Documentation of recovering time id ordering.
* ./dotenv_env: Contains env files for be loaded with dotenv, content of this folder is ignored by git.
* ./EDA: Decomented EDA experiments.
* ./ETLpipeline:
  - Default_Extract_Transform.py: The Default extract and transform pipeline that collect, extract, and transform the data and save the processed data in "./processed_data".
  - Loading_Example.ipynb: An example of using pytorch Dataset subclass (RVdataset) with pytorch dataloader for data loading.
* ./model_data: The saves model weights.
* ./NNetwork: All Neural Network model source code are kept in "./proj_mod/training.py". 
  - Decoder_transformer.ipynb: Documentation on Transformer models with both encoder and decoder layers.
  - Frozen_conv_layer.ipynb: Documentation on frozen convolution layer used to create "derivatives" of timeseries input.
  - Learned_emb_RNN: Documentation of RNN adjusted with discrete learned embedding on stock id.
  - Optimizing_hyperparameters.ipynb: Documentation of fine tuning.
  - Parameter_embedding.ipynb: Documentation of adjusting modeling with several categorical input encoded with parameter embedding on various timeseries input models.
  - RNN_with_frozen_conv.ipynb: Documentation of RNN timeseries models.
  - Transformer_with_frozen_conv_1.ipynb: Documentation of transformer models with only encoders.
* ./processed_data: The folder containing the processed data. This folder is ignored by git.
* ./proj_mod:
  - data_processing.py: The data processing functions.
  - recover_time_id.py: The time id order recovery functions.
  - training.py: Training related code, including nn.Module subclass, Dataset subclass, and training loop related code.
  - visualization.py: Code on visualization.
* ./raw_data: The folder containing the raw data. This folder is ignored by git. 
## Contributers 

Martin Molina-Fructuoso, Yuan Zhang 

## Motivation

Volatility in the stock market captures the amount of fluctuation in the stock market, and, hence, is an important quantitative indicator in the financial market. 
In this project, we seek to use data on stocks (identified with stock id) through different time id (each indicating a 10 mins time bucket) to predict the Realized Volatility in the immidiate next 10 mins after the time bucket of the time id. 
This project orignates in kaggle competition (https://www.kaggle.com/competitions/optiver-realized-volatility-prediction). 

Practicing pytorch and pandas (with sql lik querying logic) is one of the key goals of this project. 

## ETL pipeline 

One can run the py file "./ETLpipeline/Default_Extract_Transform.py" to download raw data from kaggle, and then extract and transform raw data into processed data that will be used in model training. This process contains, in order: 

* Downloading raw data from kaggle, see "./data_collecting/data_collecting_kaggle.ipynb" for detailed documentation.
* Recovering time id order, see "./data_processing/understanding_time_id_recovery.ipynb" for detailed documentation. The credit of this part goes to https://www.kaggle.com/code/stassl/recovering-time-id-order and https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/writeups/nyanp-1st-place-solution-nearest-neighbors.
* Creating Realized Volatility from raw book data, see "./data_processing/data_processing_functions.ipynb" for detailed documentation.
* Creating processed trade data from raw trade data, see "./data_processing/data_processing_functions.ipynb" for detailed documentation.
* Creating timeseries data from raw book data, see "./data_processing/data_processing_functions.ipynb" for detailed documentation.

Python pandas with sql style querying logic (e.g. merge, groupby, aggregate, window functions, and etc), and joblib parallel and delayed functions proved valuable in the process of extracting and transforming data. 

To see an example of loading the data for training use, see "./ETLpipeline/Loading_Example.ipynb". The main components include: 

* Using the RVdataset pytorch Dataset subclass to initialize a pytorch Dataset object for pytorch dataloader. See "./data_processing/create_datasets.ipynb" for detailed documention, the source code is stored in "./proj_mod/training.py".
* Loading the values for training use by feeding the prepared pytorch dataloaders to custom made training loop reg_training_loop_rmspe stored in "./proj_mod/training.py" as parameters. 

Pandas pivot is a key tool in RVdataset. 

## EDA 

## Base line model 

## Neural network models 

### Frozen convolution layer for "derivative of timeseries" feature creation 
Here we discuss the untrainable frozen convolution layer for creating "derivative of timeseries" features. 

### Timeseries based models 
Here we discuss the models that only takes timeseries as input. 

### Adjustment models 
Here we discuss the models that adjust the result produced by timesereies based models (referred as "base model" in this context) with tabular parameters that are used for parameter embedding distinguishing categories including time, stock, and row id. 

## Fine tuning 

## Future 
