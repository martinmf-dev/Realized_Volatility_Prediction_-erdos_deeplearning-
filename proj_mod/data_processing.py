import pandas as pd
import numpy as np
import glob

from joblib import Parallel, delayed
import gc

# import sklearn

def log_return(list_wap):
    return np.log(list_wap).diff() 

def rv(series_log_return): 
    return np.sqrt(np.sum(series_log_return**2))

def realized_vol(df_in,return_row_id=True): 
    """
    A function that returns the realized volatility based on the log return series input (for instance book["log_return"] where is book is a pandas dataframe containing book data for a stock and time id). 
    
    :param df_in: dataframe containing log return data (with "log_return" column). 
    :param return_row_id: Decides if the return value is a pair of form (Realized volatility,row_id) or only Realized volatility, defaulted to be True. 
    :return: return a value or a pair depending on the choice of return_row_id. 
    """
    
    series_log_return=df_in["log_return"]
    rv=np.sqrt(np.sum(series_log_return**2))
    if (return_row_id & (len(df_in)==0)):
        print("Can not harvest stock_id and time_id")
    if (return_row_id & (len(df_in)!=0)):
        stock_id=str(df_in["stock_id"].iloc[0])
        time_id=str(df_in["time_id"].iloc[0])
        row_id=stock_id+"-"+time_id
        return rv, row_id
    return rv

def create_df_wap_logreturn(df_raw_book,create_alt=True):
    #Modified 06/23/25 added create_alt parameter, if we need further new parameters in the future, we can use similar method to add them in. 
    """
    Takes in a book df and return a df of the book with wap and log return created. 
    
    :param df_raw_book: The input book df. 
    :param create_alt: Defaulted to true, create the wap and log returns according to alternative definitions as well. 
    :return: The desired df. 
    """
    # =pd.read_parquet(book_path) 
    df_raw_book["wap"]=(df_raw_book["bid_price1"]*df_raw_book["ask_size1"]+df_raw_book["ask_price1"]*df_raw_book["bid_size1"])/(df_raw_book["bid_size1"]+df_raw_book["ask_size1"])
    df_raw_book["log_return"]=df_raw_book.groupby(["time_id"])["wap"].apply(log_return).reset_index(drop=True)

    if create_alt:
        df_raw_book["wap_mid"] = (df_raw_book["bid_price1"]*df_raw_book["bid_size1"]+df_raw_book["ask_price1"]*df_raw_book["ask_size1"]\
                                 +df_raw_book["bid_price2"]*df_raw_book["bid_size2"]+df_raw_book["ask_price2"]*df_raw_book["ask_size2"])/(df_raw_book["bid_size1"]+df_raw_book["ask_size1"]+df_raw_book["bid_size2"]+df_raw_book["ask_size2"])
        df_raw_book["log_return_mid"] = (df_raw_book.groupby("time_id")["wap_mid"].transform(lambda x: np.log(x).diff()))
    return df_raw_book[~df_raw_book["log_return"].isnull()]

def create_value_for_df_by_group(df,list_gp_cols,dict_funcs,dict_rename):
    """
    A function that take a df and returns a df grouped by columns required with functions applied to the df's columns and renameds the column. 
    This function is created purely to reduced the length of a line in programing, all it does is citing df.groupby, df.agg and df.rename. 
    
    :param df: In take dataframe. 
    :param list_gp_cols: The columns of the dataframe to group by. 
    :param dict_funcs: A diction of functions to apply to chosen columns (the columns are the keys of the dict). 
    :param dict_rename: Rename chosen columns. 
    :return: A dataframe as required. 
    """
    if list_gp_cols==None: 
        df_out=pd.DataFrame(df.agg(dict_funcs)).reset_index()
        df_out=df_out.rename(columns=dict_rename)
        return df_out
    
    df_out=pd.DataFrame(df.groupby(list_gp_cols).agg(dict_funcs)).reset_index()
    df_out=df_out.rename(columns=dict_rename)
    return df_out

def create_df_RV_by_row_id(str_path): 
    """
    A function that creates a dataframe with RV organized by row_id. 
    
    :param str_path: A string of path to the parquet file of book data. 
    :return: A dataframe as describe. 
    """
    # Last modified 25/06/23
    list_parquets=glob.glob(str_path+"/*")
    df_rv=pd.DataFrame()
    for path in list_parquets: 
        df_raw_book=pd.read_parquet(path) 
        # df_raw_book["wap"]=(df_raw_book["bid_price1"]*df_raw_book["ask_size1"]+df_raw_book["ask_price1"]*df_raw_book["bid_size1"])/(df_raw_book["bid_size1"]+df_raw_book["ask_size1"])
        # df_raw_book["log_return"]=df_raw_book.groupby(["time_id"])["wap"].apply(log_return).reset_index(drop=True)
        # df_raw_book=df_raw_book[~df_raw_book["log_return"].isnull()]
        df_raw_book=create_df_wap_logreturn(df_raw_book)
        # df_rv_stock=pd.DataFrame(df_raw_book.groupby(["time_id"])["log_return"].agg(rv)).reset_index()
        # df_rv_stock=df_rv_stock.rename(columns={"log_return":"RV"})
        df_rv_stock=create_value_for_df_by_group(df_raw_book,list_gp_cols=["time_id"],dict_funcs={"log_return":rv},dict_rename={"log_return":"RV"})
        stock_id=path.split("=")[1]
        df_rv_stock["row_id"]=df_rv_stock["time_id"].apply(lambda x:f"{stock_id}-{x}")
        # Adds a column for 'stock_id'
        df_rv_stock["stock_id"]= stock_id
        # Comments out this line, as we are interested in returning all the columns
        # df_rv_stock=df_rv_stock[["row_id","RV"]]
        df_rv=pd.concat([df_rv,df_rv_stock])
    # Reindexes rows 
    df_rv = df_rv.reset_index(drop=True)
    return df_rv

def create_df_RV_by_row_id_stock(path): 
    # This function is needed for the parallelized function 'create_df_RV_by_row_id_parallel'
    # Last modified 25/06/20, 25/06/23, ....
    """
    A function that creates a dataframe with RV for ONE stock organized by row_id. 
    
    :param path: A string of path to the parquet files of book data. 
    :return: A dataframe as describe for one stock. 
    """
    df_raw_book=pd.read_parquet(path) 
    df_raw_book=create_df_wap_logreturn(df_raw_book)
    df_rv_stock=create_value_for_df_by_group(df_raw_book,list_gp_cols=["time_id"],dict_funcs={"log_return":rv},dict_rename={"log_return":"RV"})
    stock_id=path.split("=")[1]
    df_rv_stock["row_id"]=df_rv_stock["time_id"].apply(lambda x:f"{stock_id}-{x}")
    # Adds a column for 'stock_id'
    df_rv_stock["stock_id"]= stock_id
    # Comments out this line, as we are interested in returning all the columns
    #df_rv_stock=df_rv_stock[["row_id","RV"]]
    return df_rv_stock

def create_df_RV_by_row_id_parallel(str_path): 
    # Last modified 25/06/20, 25/06/23, ...
    """
    A function that, in parallel, creates a dataframe with RV organized by row_id. 
    
    :param str_path: A string of path to the DIRECTORY of parquet files of book data. 
    :return: A dataframe as describe. 
    """  
    list_parquets=glob.glob(str_path+"/*")
    # For Parallel to play well with memory allocation, the function 'create_df_RV_by_row_id_stock' needs to be defined outside this function.
    # If the memory usage spike is to big, try replacing 'n_jobs=-1' by 'n_jobs=4' or some small number
    df_rv_stock_list= Parallel(n_jobs=-1, backend='multiprocessing', batch_size=1)(delayed(create_df_RV_by_row_id_stock)(path) for path in list_parquets)
    # ignore the indexes of elements in df_rv_stock_list and reindexes the rows
    df_rv = pd.concat(df_rv_stock_list, ignore_index=True)
    return df_rv


def create_df_trade_vals_by_row_id(str_path):
    """
    A function that takes in path the trade parquet data and create trade data (avg and std of of price, size, order, and sum of size, and order) for a time bucket for each stock. 
    """
    # Last modified 25/06/23
    parquet_trade=glob.glob(str_path+"/*")
    df_vals=pd.DataFrame()
    for path in parquet_trade: 
        df_raw_trade=pd.read_parquet(path)
        df_trade_vals=df_raw_trade.groupby(["time_id"]).agg({"price":["mean","std"], "size":["sum","mean","std"], "order_count":["sum","mean","std"]}).reset_index()
        df_trade_vals.columns=df_trade_vals.columns.map("_".join)
        df_trade_vals=df_trade_vals.rename(columns={"time_id_":"time_id"})
        stock_id=path.split("=")[1]
        df_trade_vals["row_id"]=df_trade_vals["time_id"].apply(lambda x:f"{stock_id}-{x}")
        # Adds a column for 'stock_id'
        df_trade_vals["stock_id"]= stock_id
        # Comments out this line, as we want to keep 'time_id'
        # df_trade_vals.drop(columns=["time_id"],axis=1,inplace=True)
        df_vals=pd.concat([df_vals,df_trade_vals])
    # Reindexes rows 
    df_vals = df_vals.reset_index(drop=True)
    return df_vals

def book_for_stock(str_file_path,stock_id,time_id,create_para=True):
    """
    A function that returns a pandas dataframe containing the book data of a stock specified in str_file_path with certain time_id. 
    The function defaulted to create the wap and log return of wap, but this can be turned off by setting create_para to False. 
    NOTICE: THIS FUNCTION IS INEDDICIENT, BE ADVICED TO USE create_df_RV_by_row_id_stock INSTEAD. 
    
    :param str_file_path: A str of the path to the file of book data (parquet). 
    :param stock_id: The chosen stock_id. 
    :param time_id: An int that indicates the interested time_id. 
    :param create_para: Decides if wap and log return of wap will be created, defaulted to True. 
    :return: A dataframe containing book data of a chosen stock at time id. 
    """
    int_stock_id=stock_id
    str_file_path=str_file_path+"/stock_id="+str(int_stock_id)
    df_raw_book=pd.read_parquet(str_file_path)
    df_raw_book=df_raw_book[df_raw_book["time_id"]==time_id]
    # # int_stock_id=int(str_file_path.split("=")[1])
    df_raw_book.loc[:,"stock_id"]=int_stock_id
    if create_para: 
        df_raw_book["wap"] = (df_raw_book["bid_price1"]*df_raw_book["ask_size1"]+df_raw_book["ask_price1"]*df_raw_book["bid_size1"])/(df_raw_book["bid_size1"]+df_raw_book["ask_size1"])
        df_raw_book.loc[:,"log_return"] = np.log(df_raw_book["wap"]).diff()
        df_raw_book=df_raw_book[~df_raw_book["log_return"].isnull()]
    # resets the index and removes the original index
    df_raw_book=df_raw_book.reset_index(drop=True)
    
    return df_raw_book

def trade_for_stock(str_file_path,stock_id,time_id):
    """
    A function that returns a pandas dataframe containing the trade data of a stock at a chosen time_id. 
    NOTICE: THIS FUNCTION IS INEFFICIENT, BE ADVICED TO USE create_df_trade_vals_by_row_id INSTEAD.
    
    :param str_file_path: A str of the path to the file of trade data (parquet). 
    :param stock_id: The chosen stock_id. 
    :param time_id: An int that indicated the interested time_id. 
    :return: A pandas dataframe containing trade data of a chosen stock at time_id. 
    """
    int_stock_id=stock_id
    str_file_path=str_file_path+"/stock_id="+str(int_stock_id)
    df_raw_trade = pd.read_parquet(str_file_path) 
    df_raw_trade=df_raw_trade[df_raw_trade["time_id"]==time_id]
    # int_stock_id=int(str_file_path.split("=")[1])
    df_raw_trade.loc[:,"stock_id"]=int_stock_id
    return df_raw_trade 
    
def time_cross_val_split(list_time,n_split=4,percent_val_size=10): 
    """
    A function that take in a list of time id and return a time series split for cross validation, this function is written due to issue with sklearn.model_selection.TimeSeriesSplit. 
    
    :param list_time: A list of time id. 
    :param n_split: Defaulted to 4, the integer number of folds. 
    :param percent_val_size: Defaulted to 10, a float number between 0 and 100 as the percentage of the total data to be considered as test set for each fold, the function takes the floor when the necessary. 
    :return: An enumerate of values in form of (fold_index, (train_index, test_index)) where fold_index run from 0 to n_split-1. 
    """
    time_len = len(list_time)
    val_size = np.floor(time_len)*(percent_val_size/100)
    return_list=[]
    train_ends=[]
    test_ends=[]
    if n_split*percent_val_size>=100: 
        print("Please reduce either fold number or validation set size, returning empty list")
        return return_list
    for fold in range(n_split): 
        train_ends.append(int((time_len-1)-val_size*(n_split-fold)))
        test_ends.append(int((time_len-1)-val_size*(n_split-fold-1)))
        print("In fold",int(fold),":\n")
        print("Train set end at",list_time[train_ends[-1]],".\n")
        print("Test set start at",list_time[train_ends[-1]+1],"end at",list_time[test_ends[-1]],".\n")
        return_list.append((list_time[:train_ends[-1]+1],list_time[train_ends[-1]+1:test_ends[-1]+1]))
    
    # tscv = sklearn.model_selection.TimeSeriesSplit(gap=0,n_splits=n_split,test_size=val_size)
    
    # for fold_id, (train_index,test_index) in enumerate(tscv.split(list_time)):
    #     print("In fold",int(fold_id),"\n")
    #     print("Train set start at",train_index[0],"and end at",train_index[-1],"\n")
    #     print("Test set start at",test_index[0],"end at",test_index[-1])
    # return enumerate(tscv.split(list_time))

    return enumerate(return_list)

# def create_RV_timeseries(df_in, n_subint=60, in_start=0, in_end=600): 
#     """
#     A function that created RV for subintervals of the whole interval for a chosen stock_id and time_id. 
#     It is expected that (in_end-in_start)%n_subint==0. 
    
#     :param df_in: A pandas dataframe with "log_return" and "seconds_in_bucket" columns. 
#     :param n_subint: Integer number of total intervals wanted, defaulted to 60. 
#     :param in_start: Integer start of the start of time interval, defaulted to 0. 
#     :param in_end: Integer end of the end of time interval, defaulted to 600. 
#     :return: A numpy array of length n_subint populated by RV of each sub-interval. 
#     """
    
#     in_len=in_end-in_start
#     # print("length of interval is",in_len)
#     if (in_len%n_subint): 
#         # print("Both the length of interval and the total number of sub-interval are expected to be integers. The length of interval is not divisible by number of subinterval, returning None.")
#         return None
#     int_sublen=int(in_len/n_subint)
#     arr_RV=np.zeros(n_subint)
#     # print("length each sub-interval is", int_sublen)
#     for index in range(n_subint):
#         subin_start=index*int_sublen
#         subin_end=(index+1)*int_sublen
#         subin_RV=realized_vol(df_in=df_in[(df_in["seconds_in_bucket"]>subin_start)&(df_in["seconds_in_bucket"]<=subin_end)],return_row_id=False)
#         # print("RV of the sub-interval",(index+1),"is",subin_RV)
#         arr_RV[index]=subin_RV
#         # print(arr_RV)
#     return arr_RV

def create_timeseries_stock(path, dict_agg, dict_rename, n_subint=60, in_start=0, in_end=600, create_row_id=True): 
    #Created 25/06/23   
    #Modified 25/06/24 Fixed issue where some time id is missing sub intervals. 
    """
    A function that creates a df with desired time series features of the whole interval for a chosen stock_id. 
    It is expected that (in_end-in_start)%n_subint==0. 
    
    :path: A string of path to the parquet files of book data. 
    :param dict_agg: A dictionary in form of {string indicating the column \: the function to apply to the column,...}. As an example, {"log_return" \: data_processing.rv} will create the RV time serie for each of the time_id for the chosen stock id. Notice that only one function is allowed for one column. 
    :param dict_rename: A dictionary in form of {string indicating the column \: the new name of the column,...}. As an example, {"log_return" \: "sub_int_RV"} will name the newly created time series feature according to column "log_return" as "sub_int_RV". 
    :param n_subint: Integer number of total intervals wanted, defaulted to 60. 
    :param in_start: Integer start of the start of time interval, defaulted to 0. 
    :param in_end: Integer end of the end of time interval, defaulted to 600. 
    :param create_row_id: Defaulted to True, decides if the row id will be created. 
    :return: The desired df. 
    """
    # Constructs dataframe with book data for the stock specified in path
    df_in = pd.read_parquet(path)
    df_in['stock_id']=int(path.split("=")[1])
    df_in=create_df_wap_logreturn(df_in)
    df_in["time_id"]=df_in["time_id"].astype(int)
    in_len=in_end-in_start
    stock_id=df_in["stock_id"].iloc[0]
    # print("length of interval is",in_len)
    if (in_len%n_subint): 
        print("Both the length of interval and the total number of sub-interval are expected to be integers. The length of interval is not divisible by number of subinterval, returning None.")
        return None
    int_sublen=int(in_len/n_subint)
    df_out=pd.DataFrame()
    op_columns=list(dict_agg.keys())
    time_id_list=df_in["time_id"].unique()
    for index in range(n_subint):
        subin_start=index*int_sublen
        subin_end=(index+1)*int_sublen
        df_step=df_in[(df_in["seconds_in_bucket"]>subin_start)&(df_in["seconds_in_bucket"]<=subin_end)].groupby("time_id", dropna=False).agg(dict_agg).reset_index()
        df_step["time_id"]=df_step["time_id"].astype(int)
        time_id_present=df_step["time_id"].unique()
        time_id_missing=[time for time in time_id_list if time not in time_id_present]
        df_patch=pd.DataFrame({"time_id":time_id_missing})
        df_patch[op_columns]=0
        df_step=pd.concat([df_step,df_patch])
        del df_patch
        df_step=df_step.rename(columns=dict_rename)
        df_step["sub_int_num"]=index+1
        df_step["stock_id"]=int(stock_id)
        if create_row_id:
            df_step["row_id"]=df_step["stock_id"].astype(int).astype(str)+"-"+df_step["time_id"].astype(int).astype(str)
        df_out=pd.concat([df_out,df_step])
        del df_step
        df_out["time_id"]=df_out["time_id"].astype(int)
        # df_out["row_id"]=df_out["row_id"].apply(lambda x: x.split(".")[0])
        # print("finished for sub interval",index+1)
    # resets the index and removes the original index
    df_out=df_out.reset_index(drop=True)
    # Deletes intermediate dataframe to free up RAM
    del df_in
    gc.collect()
    return df_out


def create_timeseries(str_path, dict_agg, dict_rename, n_subint=60, in_start=0, in_end=600, create_row_id=True):
    # Created 25/06/23
    # Last modified 25/06/24
    """
    A function that creates a df with the time series of desired features on the whole interval for all the stocks.
    
    :str_path: A string of path to the DIRECTORY of parquet files of book data. 
    :param dict_agg: A dictionary in form of {string indicating the column \: the function to apply to the column,...}. As an example, {"log_return" \: data_processing.rv} will create the RV time serie for each of the time_id for the chosen stock id. Notice that only one function is allowed for one column. 
    :param dict_rename: A dictionary in form of {string indicating the column \: the new name of the column,...}. As an example, {"log_return" \: "sub_int_RV"} will name the newly created time series feature according to column "log_return" as "sub_int_RV". 
    :param n_subint: Integer number of total intervals wanted, defaulted to 60. 
    :param in_start: Integer start of the start of time interval, defaulted to 0. 
    :param in_end: Integer end of the end of time interval, defaulted to 600. 
    :param create_row_id: Defaulted to True, decides if the row id will be created. 
    :return: The desired df. 
    
    """
    list_parquets=glob.glob(str_path+"/*")
    df_time_series_list= Parallel(n_jobs=-1,
                                  backend='multiprocessing', 
                                  batch_size=1)(delayed(create_timeseries_stock)
                                                (path, dict_agg=dict_agg, dict_rename=dict_rename, n_subint=n_subint, in_start=in_start, 
                                                 in_end=in_end, create_row_id=create_row_id) 
                                                for path in list_parquets)
    df_time_series = pd.concat(df_time_series_list, ignore_index=True)
    return df_time_series


