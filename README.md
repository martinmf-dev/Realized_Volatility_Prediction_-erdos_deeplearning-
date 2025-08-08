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
  - Loading_Example.ipynb: An example of using pytorch Dataset subclass with pytorch dataloader for data loading.
* ./model_data: The saves model weights.
* ./NNetwork: All Neural Network model source code are kept in "./proj_mod/training.py". 
  - Decoder_transformer.ipynb: Documentation on Transformer models with both encoder and decoder layers.
  - Frozen_conv_layer.ipynb: Documentation on frozen convolution layer used to create "derivatives" of timeseries input.
  - Learned_emb_RNN: Documentation of RNN adjusted with discrete learned embedding on stock id.
  - Optimizing_hyperparameters.ipynb: Documentation of fine tuning.
  - Parameter_embedding.ipynb: Documentation of adjusting modeling with several categorical input encoded with parameter embedding on various timeseries input models.
  - RNN_with_frozen_conv.ipynb: Documentation of RNN timeseries models.
  - Transformer_with_frozen_conv_1.ipynb: Documentation of transformer models with only encoders.
* ./processed_data: The folder containing the processed data. This folder is ingored by git.
* ./proj_mod:
  - data_processing.py: The data processing functions.
  - recover_time_id.py: The time id order recovery functions.
  - training.py: Training related code, including nn.Module subclass, Dataset subclass, and training loop related code.
  - visualization.py: Code on visualization.
* ./raw_data: The folder containing the raw data. 
## Contributers 
Martin Molina-Fructuoso, Yuan Zhang 

## Motivations 
Volatility in the stock market captures the amount of fluctuation in the stock market, and, hence, is an important quantitative indicator in the financial market. 
In This project, we seek to use data on stocks (identified with stock id) through different time id (each indicating a 10 mins time bucket) to predict the Realized Volatility in the immidiate next 10 mins after the time bucket of the time id. 

## ETL pipeline 

## EDA 

## Base line model 

## Neural network models 

## Future 
