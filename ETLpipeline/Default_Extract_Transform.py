import subprocess
import sys
import numpy as np
import time
import pandas as pd

sys.path.append("../")
from proj_mod import data_processing, recover_time_id

def Default_Ext_Trans():
    #Getting raw data
    get_raw=input("Would you like to download raw data from kaggle?(Y/N)").upper()
    if get_raw=="Y": 
        print("Retrieving raw data from kaggle: \n")
        subprocess.call(["python data_collecting_kaggle.py"],cwd="./data_collecting/", shell=True)
        print("Raw data received from kaggle. Stored in \"./raw_data/kaggle_ORVP\". Contents of this folder will be ignored by git.\n")
    
    #Recoverying time
    recover_time=input("Would you like to recover time id order?(Y/N)").upper()
    if recover_time=="Y": 
        print("Recovering time id ordering...\n")
        recovered_time_id_order_list = recover_time_id.reconstruct_time_id_order(str_path="../raw_data/kaggle_ORVP/book_train.parquet")
        recovered_time_id_order = np.array(recovered_time_id_order_list, dtype=int)
        print("Time id order recovered. Starting to save it. \n")
        np.save('../processed_data/recovered_time_id_order.npy',recovered_time_id_order)
        print("Recovered time id order saved to \"../processed_data/recovered_time_id_order.npy\". Contents of this folder will be ignored by git.\n")
        
    #Creating data with default set ups. 
    create_data=input("Would you like to transform raw data into ready-to-use data?(Y/N)").upper()
    if create_data=="Y": 
        print("Transforming data...\n")
        
        print("Creating RV by row_id ...\n")
        path_book="../raw_data/kaggle_ORVP/book_train.parquet"
        start_time = time.time()
        df_rv=data_processing.create_df_RV_by_row_id_parallel(path_book)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Created RV by row id data if {elapsed_time} seconds.\n")
        df_rv.to_parquet("../processed_data/RV_by_row_id.parquet",index=False)
        print("Saved to \"../processed_data/RV_by_row_id.parquet\". Contents of this folder will be ignored by git.\n")
        save_RV_csv=input("Would you like to save it also as csv?(Y/N)").upper()
        if save_RV_csv=="Y": 
            df_rv.to_csv("../processed_data/RV_by_row_id.csv",index=False)
            print("Saved to \"../processed_data/RV_by_row_id.csv\". Contents of this folder will be ignored by git.\n")
        
        print("Creating trade values by row id ...\n")
        start_time = time.time()
        df_tab=data_processing.create_df_trade_vals_by_row_id(str_path="../raw_data/kaggle_ORVP/trade_train.parquet")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Created trade values by row id data in {elapsed_time} seconds.")
        df_tab.to_parquet("../processed_data/trade_vals_by_row_id.parquet",index=False)
        print("Saved to \"../processed_data/trade_vals_by_row_id.parquet\". Contents of this folder will be ignored by git.\n")
        save_trade_csv=input("Would you like to save it also as csv?(Y/N)").upper()
        if save_trade_csv=="Y": 
            df_tab.to_csv("../processed_data/trade_vals_by_row_id.csv",index=False)
        
        print("Creating trade values grouped by time id and row id separately respectively ...")
        start_time = time.time()
        df_total=data_processing.create_df_param_emb_by_group(df_in=df_tab,str_groupby="time_id",dict_params_aggfun={"price_mean":["mean","std"],"size_sum":["mean","std"],"order_count_sum":["mean","std"]})
        df_total=data_processing.create_df_param_emb_by_group(df_in=df_total,str_groupby="stock_id",dict_params_aggfun={"price_mean":["mean","std"],"size_sum":["mean","std"],"order_count_sum":["mean","std"]})
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Created trade values grouped by time and stock in {elapsed_time} seconds.")
        df_total.to_parquet("../processed_data/total_trade_values.parquet",index=False)
        print("Saved to \"../processed_data/total_trade_values.parquet\". Contents of this folder will be ignored by git.\n")
        save_total_csv=input("Would you like to save it also as csv, this file will be large (hence, advised against)?(Y/N)").upper()
        if save_total_csv=="Y":
            df_total.to_csv("../processed_data/total_trade_values.csv",index=False)
            
        print("Creating timeseries data...")
        start_time = time.time()
        path_book="../raw_data/kaggle_ORVP/book_train.parquet"
        book_time=data_processing.create_timeseries(path_book,dict_agg={"log_return":data_processing.rv},dict_rename={"log_return":"sub_int_RV"})
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Created timeseries data in {elapsed_time:.2f} seconds.")
        book_time.to_parquet("../processed_data/book_RV_ts_60_si.parquet")
        print("Saved to \"../processed_data/book_RV_ts_60_si.parquet\". Contents of this folder will be ignored by git.\n")
        save_time_csv=input("Would you like to save it also as csv, this file will be VERY large (hence, HIGHLY advised against)?(Y/N)").upper()
        if save_time_csv=="Y": 
            book_time.to_csv("../processed_data/book_RV_ts_60_si.csv")
        
    return 0
    
print("In this process, raw data will be downloaded from kaggle. Afterwhich, default relevant data will be extracted and transformed with python pandas code with SQL like queries which will then be safed on file.\n")
start=input("Start the process?(Y/N)\n").upper()
if start=="Y": 
    print("If this is the first time you run this code, please input Y to all options. Operation starting ...\n")
    Default_Ext_Trans()
    print("Process finished. Please keep in mind that, due to the structure of pytorch, the loading procedure will be done through training.RVdataset to create pytorch Dataset subclass so that it can be loaded into training loop with pytorch dataloader.")
else: 
    print("Exiting the process")

