# Realized Volatility Prediction 

## Folder and file organization (fill in)
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

## Folder and file explanations (fill in)

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

Martin Molina-Fructuoso (https://github.com/martinmf-dev), Yuan Zhang (https://github.com/YCoeusZ)

## Motivation

Volatility in the stock market captures the amount of fluctuation in the stock market, and, hence, is an important quantitative indicator in the financial market. 
In this project, we seek to use data on stocks (identified with stock id) through different time id (each indicating a 10 mins time bucket) to predict the Realized Volatility in the immidiate next 10 mins after the time bucket of the time id. 
This project orignates in kaggle competition (https://www.kaggle.com/competitions/optiver-realized-volatility-prediction). 

Practicing pytorch and pandas (with sql lik querying logic) is one of the key goals of this project. 

## Necessary packages 
The following are absolutely necessary python packages needed for this project (extreme common packages like sys, os, numpy, and so on will not be listed) (fill in): 
* kaggle (for downloading raw data)
* pandas (for data extraction and transformation)
* pytorch (for data loading, model creation, and training; Depending on your GPU hardware, the detailed set up of pytorch, cuda, and (or) rocm might alter greatly)
* optuna (for fine tuning)
* sklearn, statsmodel, pingouin, scipy (for EDA, and base line model)
* matplotlib (for visualization)
* joblib (for parallel computing) 

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

## Base line model 
(fill in)

## Neural network models 
The following is a summary of the models in their default settings, many of the models, in reality, offers much more flexibility to be altered. 

---
### Frozen convolution layer for "derivative of timeseries" feature creation 
Here we discuss the untrainable frozen convolution layer for creating "derivative of timeseries" features. 
This layer applies tensor (-1,1) to produce "derivative feature" for a timeseries (and append 0 at the end). 
As an example, input (1.,  2.,  3.,  1.,  2.,  3.,  1.,  2.,  3.) will have derivative (1.,  1., -2.,  1.,  1., -2.,  1.,  1.,  0.). 
This process can be applied for several time to produce "$`n^{th}`$ derivative feature". 

<img width="287" height="165" alt="image" src="https://github.com/user-attachments/assets/1a3fe006-1011-4dea-8a64-cfb065d0edb5" />

This is a key layer that is used in all models to produce timeseries with "derivative" values. 
See detail at "./NNetwork/Frozen_conv_layer.ipynb" for documentation.  

---
### Transformer building blocks  
Here we discuss the custom transformer based building blocks. 

* **Encoder**

A custom encoder layer: 

<img width="386" height="458" alt="image" src="https://github.com/user-attachments/assets/65eeaa29-04cb-4001-b65c-abc81c872e4d" />

Source code ts_encoder at "./proj_mod/training.py". 

* **Decoder**

A custom decoder layer: 

<img width="480" height="615" alt="image" src="https://github.com/user-attachments/assets/a71ef29d-9ea5-4e2c-bf07-4b20f3767320" />

Source code ts_decoder at "./proj_mod/training.py". 

* **Positional embedding by cross attention**

A custom postional embedding layer to preserve positional signal in ordered input: 

<img width="469" height="285" alt="image" src="https://github.com/user-attachments/assets/d65faf78-61b1-4cdb-85d0-8f8cab221594" />

Source code pos_emb_cross_attn at "./proj_mod/training.py". 
A positional embedding is necessary because the property $`Attention(AQ,BK,BV)=A\ Attention(Q,K,V)`$, intuitively, this means that attention layer is "permutation equivariant in respect to rows of Q", meaning changing the order of element of elements in timeseries input at Q does not change the output. So we will need to keep the signal of position somehow. See a detailed reasonaing at "./NNetwork/Transformer_wtih_frozen_conv_1.ipynb". 

---

### Timeseries based models 
Here we discuss the models that only takes timeseries as input. 

#### RNN timeseries based model 
We first discuss the rnn based model for timesieres input: 

<img width="397" height="530" alt="image" src="https://github.com/user-attachments/assets/340d3ad1-4590-43b9-a228-882fd16f3ede" />

Best loss for rnn: (fill in); Best loss for lstm: (fill in); Best loss for gru: (fill in)

Source code RV_RNN_conv at "./proj_mod/training.py". See detailed decumentation at "./NNetwork/RNN_with_frozen_conv.ipynb". 

#### Transformer timeseries based models 
We now discuss the transformer based modle for timeseries input. 

* **Encoder only transformer**

The first mode is an encoder only transformer:

<img width="256" height="416" alt="image" src="https://github.com/user-attachments/assets/2e0d63ea-52fb-4840-9433-4faf34e9482a" />

Best loss: (fill in)

Source code encoder_ensemble at "./proj_mod/training.py". See detailed documentation at "./NNetwork/Transformer_with_frozen_conv_1.ipynb". 

* **Encoder decoder teacher forcing transformer**

The following is a transformer with both encoder and decoder, we will use the self attention encoder ouput of the input timesereis as the ground target "teacher" (since we do not have a connecting timeseries): 

<img width="234" height="562" alt="image" src="https://github.com/user-attachments/assets/2c38d43d-8cae-4466-8daa-129201fe84c5" />

Best loss: (fill in)

Source code encoder_decoder_teacherforcing at "./proj_mod/training.py". See detailed documentation at "./NNetwork/Decoder_transformer.ipynb", where an explanation on casual masking is detailed as well. 

---

### Adjustment models 
Here we discuss the models that adjust the result produced by timesereies based models (referred as "base model" in this context) with tabular parameters that are used for parameter embedding distinguishing categories including time, stock, and row id. 

#### Adjustment with only stock id (discrete learned embedding): 
As a proof of concept, we first limited to only adjusting with the stock id with discrete learned embedding. 
To acheive discrete learned embedding efficiently, we first created emb id (i.e. embedding id) to replace the stock id. 
The only difference between emb id and stock id is that embd id is a list of integer with no "gape" while stock id does jump over some integers. 

* Adjustment by pre-appending

We simply pre-appended the embedded emb id infront of the timeseries: 

<img width="502" height="409" alt="image" src="https://github.com/user-attachments/assets/ba193a74-bfc9-427e-ae7b-8fb2d37c345f" />

Best loss: (fill in)

See detailed documentation at "./NNetwork/Learned_emb_RNN.ipynb". 

* Adjustment by multiplication

We have a sub network that works on the embedded emb id to create a scalar adjuster: 

<img width="502" height="328" alt="image" src="https://github.com/user-attachments/assets/570ab36c-4bc8-4f27-82a1-2c1a31ffed7c" />

Best loss: (fill in)

Source code id_learned_embedding_adj_rnn_mtpl at "./proj_mod/training.py". See detailed documentation at "./NNetwork/Learned_emb_RNN.ipynb". 

* An adjustment with cross attention

We have a sub model produce a vector pre-adjuster which we will use as Key and Value in a cross attention layer with the base model output as the Query, the output of this cross attention layer is then used as the adjuster: 

<img width="551" height="694" alt="image" src="https://github.com/user-attachments/assets/55a87e7a-6459-41c5-bbf5-214b9e92cb6e" />

Best loss: (fill in) Using base model: (fill in) 

Source code class id_learned_embedding_attend_rnn at "./proj_mod/training.py". See detailed documentation at "./NNetwork/Learned_emb_RNN.ipynb". 

#### Adjustment with row id, stock id, and time id 
We constructed a model that has capibility to adjust the timeseries base model output with any subset of row id, stock id, time id, and emb id: 

<img width="711" height="619" alt="image" src="https://github.com/user-attachments/assets/47ea76ca-d15b-4542-b3c0-6acaad9f19ad" />

Best loss: (fill in) Using base model: (fill in) 

Source code class multi_adj_by_attd at "./proj_mod/training.py". See detailed documentation at "./NNetwork/Parameter_embedding.ipynb". 

## Fine tuning 
We fine-tuned two encoder-decoder modles, with and without a stock identifier embedding. The optimization strategy and heuristically defined search space, both critical to making Optuna work effectviely, were developted using domain knowledge and targeted experimentation. We used the Optuna library to methodically explore the hyperparameter space combined with custom early termination of undeperforming trials. In addition, we designed reusable functions and a modular structure so the same optimization pipeline can be applied to any custom model, ensuring that the optimization strategy is transparent, models are self-contained, and reproducibility of models for inference is guaranteed.


## Future 
As of August 10th of 2025, this project is 3 month in age. Both contributors have honed their skills and understanding in data transformation and machine learning with pytorch. 
Both contributors feel that they are just getting started and there are many things to do to keep improving the models (fill in): 
* As of now, most models uses feed forward layers mostly composed of custom made encoder and decoders, or nn.Linear. The contributors want to investigate further into useing certain alternatives including convolutions, especially for the data with larger dimensions.
* As of now, the model that is adjusted with all of row id, stock id, time id, and emb id is over training very fast (Although, it has very good validation loss). This indicates bad regularization, and possibly noizy input parameters, the contributers want to investigate into implimenting methods to change this: increasing dropout, changing weight decay of optimizer (kind of an analog of ridge (L2) regression for our context), methods to reduce input parameters (like lasso (L1) regression in the context of linear regression), and so on. 
* Adjusting the teacher forcing model further: Currently, the teacher forcing model uses the encoder output as both the encoder memory and the ground target (the "teacher"), the contributers want to investigate into cutting the input timeseries in the middle, and use the first as "true input" and the second half as the "teacher".
* Investigate further into fine tuning methods.
* Find improved methods for processing the data: pandas is a great tool, but the contributers have noticed its speed issue when haddling massive amount of data, even when running under parallel. So contributors want to investigate further into otherwise methods to improve this, for instance, using tools that are designed for big data (for instance, pyspark, and so on).
* Currently all the adjustment models are training the timeseries base models as "submodels", we also would like to investigate into using a pretrained base model output as a parameter to be adjusted on. 

## Citations 
* Attention is all you need: https://arxiv.org/pdf/1706.03762
* Kaggle competition: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction
* Kaggle competition top solution: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/writeups/nyanp-1st-place-solution-nearest-neighbors
* Kaggle forum recovering time id order: https://www.kaggle.com/code/stassl/recovering-time-id-order
