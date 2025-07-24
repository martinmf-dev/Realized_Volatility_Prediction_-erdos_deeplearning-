import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
import pandas as pd
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset, DataLoader

#Metric###############################################################################################################

def rmspe(y_true, y_pred):
    '''
    Compute Root Mean Square Percentage Error between two arrays.
    '''
    loss = np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true)), axis=0))

    return loss

def rmspe_of_linear(x,y): 
    """
    Find rmspe when predicting y with x by linear regression. 
    """
    model=LinearRegression()
    model.fit(x,y)
    pred_linear=model.predict(x)
    return rmspe(y_pred=pred_linear,y_true=y)

def rmspe_linear_df(df,list_feature,str_target):
    """
    Find rmspe when predicting target column of dataframe with features in dataframe by linear regression. 
    """
    return rmspe_of_linear(df[list_feature],df[[str_target]])

class RMSPELoss(nn.Module): 
    #Created 06/25/25 
    """
    RMSPE is not a premade pytorch loss function, the following creates the loss function so that it can be used in training, if needed. 
    
    :param eps: Defaulted to 0. The small value needed to avoid division by zero. 
    """
    def __init__(self,eps=0): 
        super(RMSPELoss,self).__init__()
        self.eps=eps
        
    def forward(self, ypred, ytrue): 
        return torch.sqrt(torch.mean(torch.square((ypred-ytrue)/(ytrue+self.eps))))
    
class MSPELoss(nn.Module): 
    #Created 06/25/25 
    """
    Mean squared percentage error is not a premade pytorch loss function, the following creates the loss function so that it can be used in training, if needed. 
    
    :param eps: Defaulted to 0. The small value needed to avoid division by zero. 
    """
    def __init__(self,eps=0): 
        super(RMSPELoss,self).__init__()
        self.eps=eps
        
    def forward(self, ypred, ytrue): 
        return torch.mean(torch.square((ypred-ytrue)/(ytrue+self.eps)))
    
#Training loop###################################################################################################################################
     
def reg_validator_rmspe(model, val_loader, device, eps=0,scaler=1, norm_train_target=False, train_target_mean=None, train_target_std=None): 
    #Created 06/25/25 In progress, testing needed. 
    """
    Returns the rmspe on the validation set for regression type training. As a reminder, one should not apply normalization to validation dataset's target, but one should apply the same normalization to the input features as the training dataset. 
    
    :param model: The model used. 
    :param val_loader: The loader that feeds the validation set. 
    :param eps: Defaulted to 0. The small value needed to avoid division by zero. 
    :param device: The device used to calculate. 
    :param scaler: Defaulted to 1. Scaling the input and output value so that they are not too small. 
    :param norm_train_target: Defaulted to False. Set to true if the training targets feed by train loader are post normalized. 
    :param train_target_mean: Defaulted to None. The mean of training target. 
    :param train_target_std: Defaulted to None. The std of training std. 
    :return: The rmspe on the validation set. 
    """
    sum_of_square=0
    total_count=len(val_loader.dataset)
    if total_count==0: 
        print("There is nothing in the validation set, returning None")
        return None
    with torch.no_grad():
        for feature, target in val_loader: 
            feature*=scaler
            target*=scaler
            feature=feature.to(device=device)
            target=target.to(device=device)
            pred=model(feature)
            if norm_train_target:
                pred=pred*train_target_std+train_target_mean
            sum_of_square+=torch.sum(torch.square((pred-target)/(target+eps)))
        rmspe=torch.sqrt(sum_of_square/total_count)
    return rmspe
        
def reg_training_loop_rmspe(optimizer, model, train_loader, val_loader, device, ot_steps=100, recall_best=True, eps=0, list_train_loss=None, list_val_loss=None, report_interval=20, n_epochs=1000, scaler=1, norm_train_target=False, train_target="target"): 
    #Created 06/25/25 In progress, testing needed
    #Modified 06/08/25 Denormalization was moved before loss calculation
    #Modified 07/23/25 Add printing best validation when updated 
    """
    A training loop for regression type training with rmspe loss function. 
    
    :param n_epochs: Defaulted to 1000. Total number of epochs. 
    :param optimizer: Optimizer wanted. 
    :param model: Model used. 
    :param train_loader: Training loader used. 
    :param val_loader: Validation loader used. 
    :param ot_steps: Defaulted to 100. The number of epochs, where, if validation loss does not improve, the training will be stopped. Turn off this feature by setting it to None. 
    :param recall_best: Defaulted to True. Reloads the model to the best version according to validation loss. 
    :param device: GPU or CPU, choose your poison. 
    :param eps: Defaulted to 0. The small value needed to avoid division by zero. 
    :param list_train_loss: Defaulted to None. If set to certain list, the function will append the training loss values to the end of the list in order of epochs. 
    :param list_val_loss: Defaulted to None. If set to certain list, the function will append the validation loss values to the end of the list in order of epochs. 
    :param report_interval: Defaulted to 20. The training loop will report once every report interval number of epochs. 
    :param scaler: Defaulted to 1. Scaling the input and output value so that they are not too small. This is helpful when debugging. 
    :param norm_train_target: Defaulted to False. Set to true if the targets feed by train loader are post normalized, in which case, it is expected that the dataset of train loader should have access to feat_norm_dict object variables. 
    :param train_target: Defaulted to "target". The string name of target in the train dataset input dataframe. 
    :return: The state dictionary of the best model, according to validation loss. 
    """
    total_data_count=len(train_loader.dataset)
    best_val_loss=0
    best_val_epoch=0
    start_time=time.time()
    # if recall_best: 
    best_mode_state_dict=dict()
    for epoch in range(1,n_epochs+1): 
        sum_of_sqaure_train=0
        #Training loop for a batch#############
        #######################################
        for feature, target in train_loader: 
            #Try to scale the values a little. 
            with torch.no_grad():
                feature*=scaler
                target*=scaler
            #First, the standard training steps 
            feature=feature.to(device=device)
            target=target.to(device=device)

            
            #Moved here
            #Create variables for training dataset target mean and std 
            train_target_mean=None
            train_target_std=None  
            
            #Moved here
            if norm_train_target:           
                train_target_mean=train_loader.dataset.feat_norm_dict[train_target][0]
                train_target_std=train_loader.dataset.feat_norm_dict[train_target][1]

            
            pred=model(feature)

            
            #Moved here
            if norm_train_target:
                pred=pred*train_target_std+train_target_mean
                target=target*train_target_std+train_target_mean
            
            #Debug print 
            # print(pred)

            
            loss_fnc=RMSPELoss(eps=eps)
            loss_step=loss_fnc(pred,target)
            #Update using optimizer according to loss of the batch
            optimizer.zero_grad()
            loss_step.backward()
            optimizer.step()
            # #Create variables for training dataset target mean and std 
            # train_target_mean=None
            # train_target_std=None            
            #Update the sum of sqaure (without grad) 
            with torch.no_grad():
                # This was moved before the loss calculation
                # if norm_train_target: 
                #     train_target_mean=train_loader.dataset.feat_norm_dict[train_target][0]
                #     train_target_std=train_loader.dataset.feat_norm_dict[train_target][1]
                #     pred=pred*train_target_std+train_target_mean
                #     target=target*train_target_std+train_target_mean
                sum_of_sqaure_train+=torch.sum(torch.square((pred-target)/(target+eps)))
        #End of training loop for a batch######
        #######################################
        curr_time=time.time()
        time_cost=curr_time-start_time
        #Calculate the training loss of this epoch (without grad)
        with torch.no_grad():    
            epoch_train_loss=torch.sqrt(sum_of_sqaure_train/total_data_count)
        #Calculate the validation loss of this epoch (without grad, citing another function)
        epoch_val_loss=reg_validator_rmspe(model=model,val_loader=val_loader,eps=eps,device=device,scaler=scaler,norm_train_target=norm_train_target,train_target_mean=train_target_mean,train_target_std=train_target_std)
        #Update the best validation loss and the epoch that it occurred     
        if ((epoch==1) or (epoch_val_loss<best_val_loss)): 
                best_val_loss=epoch_val_loss
                best_val_epoch=epoch
                #Remember the best model (based on validation loss), store the state dictionary 
                print("A new best validation loss at epoch ", best_val_epoch, " with validation loss of ", best_val_loss)
                # if recall_best:
                best_mode_state_dict=model.state_dict()
        #Update the list of train and validation loss, if requested 
        if list_train_loss!= None: 
            list_train_loss.append(epoch_train_loss)
        if list_val_loss!=None: 
            list_val_loss.append(epoch_val_loss)
        #Print report according to report interval
        if ((epoch==1) or (epoch%report_interval==0)):
            print("At ", time_cost, " epoch ",epoch, "has training loss ", epoch_train_loss, " and validation loss ", epoch_val_loss,".\n")
        #If a over training stopper is requested and the model is over trained based on non-improving validation loss for a certain number of epochs, we stop the training. 
        if ((ot_steps!=None) and (epoch-best_val_epoch>=ot_steps)): 
            print("The validation loss has not improved for ",ot_steps, " epochs. Stopping current training loop.\n")
            #If requested, reload the best model (according to the best validation loss). 
            if recall_best: 
                model.load_state_dict(best_mode_state_dict)
                print("Best model state dictionary of this training loop is reloaded.\n") 
            print("According to validation loss, the best model is reached at epoch", best_val_epoch, " with validation loss: ",best_val_loss,".\n","The total number of epoch trained is ", epoch, ".\n","Training completed in: ", time_cost,".\n")
            return best_mode_state_dict
    print("All ",n_epochs," epochs have been completed.\n")
    print("According to validation loss, the best model is reached at epoch", best_val_epoch, " with validation loss: ",best_val_loss,".\n")
    #If requested, reload the best model (according to the best validation loss). 
    if recall_best: 
        model.load_state_dict(best_mode_state_dict)
        print("Best model state dictionary of this training loop is reloaded.\n")
    print("Training completed.",time_cost,"\n")
    return best_mode_state_dict

#Dataset##################################################################################################################################################

class RVdataset(Dataset): 
    #Created 06/27/25, see create_datasets.ipynb for documentation. 
    #Modified 06/30/25, added query_str option. 
    # Modified 07/02/25 added get_row_id method and numeric ordering option
    #Modified 07/02/25 fixed issue with error when tab_feature is None
    #Modified 07/03/25 added normalization functionality 
    #Modified 07/14/25 Added self.featureplace to help with spliting feature tensor. 
    #Modified 07/21/25 Now to function can auto cast strings into int for stock and time id 
    def __init__(self, query_str=None, query_val_list=None, time_id_list=None, stock_id_list=None, tab_features=None, ts_features=None, target="target", df_ts_feat=None, df_tab_feat=None, df_target=None, numeric=False, norm_feature_dict=None):
        """
        Object in subclass of Dataset. It is ADVISED to cast stock and time id as int before running this function, especially when using query_str. 
        
        :param query_str: Defaulted to None, in which case, filter to dataset data will be applied through time_id_list and stock_id_list. Set to query string to apply filter to data included in dataset, when value is not None, the values of time_id_list and srock_id_list are practically ignored.         
        :param query_val_list: Defaulted to None. The list of variables that serves as reference for query string. This parameter is practically ignored when query_str is None. 
        :param time_id_list: Defaulted to None, in which case ALL time_id's will be included. A list (numpy array will NOT work) containing the time_id's of interest. If the value is not None, it is expected that "time_id" column (with values type int) is present in all input dataframes. 
        :param stock_id_list: Defaulted to None, in which case ALL stock_id's will be included. A list (numpy array will NOT work) containing the stock_id's of interst. If the value is not None, it is expected that "stock_id" column (with values type int) is present in all input dataframes. 
        :param tab_features: Defaulted to None, in which case NO feature will be included. A list containing the string of names of columns in df_tab_feat to be used as tabular features, for instance, the RV of current 10 mins bucket is a tabular feature. 
        :param ts_features: Defaulted to None, in which case NO feature will be included. A list containing the string of names of columns in df_ts_feat to be used as time series features, for instance, sub_int_RV in book_time created in data_processing_functions.ipynb. 
        :param target: Defaulted to "target". The string indicating how target is identified in column index of dataframe. 
        :param df_ts_feat: Defaulted to None. The dataframe containing the time series like features, must have "row_id" as identifier for rows and column "sub_int_num" as indicator of time series ordering. 
        :param df_tab_feat: Defaulted to None. The dataframe containing the tabluar features, must have "row_id" as identifier for rows. When df_target is not None, one should make sure there is no target in the df_tab_feat. 
        :param df_target: Defaulted to None, in which case, target will be searched in df_tab_feat instead and expects df_tab_feat to contain target column to be used as target. The dataframe containing the target stored in the target column, must have "row_id" to be used as identifier. 
        :param numeric: Defaulted to False. When set to true, RVdataset returns rows ordered numerically first by stock_id, and then by time_id 
        :param norm_feature_dict: Defaulted to None. A dict that indicate the features one want to normalize, this can include timeseries feature (normalized accross all sub_int_num and all row_id), tabular features, and the target. The dictionary should be in form of {string of feature name: (mean, std), ...}, mean and std "forced", but if automatic calulation of mean and std is desired, replace the tuple with None.
        
        Object attributes: 
        
            self.features: The collection of features as a torch tensor. 
            self.target: The collection of targets as a torch tensor. 
            self.len: The length of the whole dataset object. 
            self.featuresplit: A dictionary in form of {feature name:length of feature, ...} to help distinguish different features in the feature torch tensor. The length of feature is included since some of the features are time series, while some are tabular. 
            self.featureplace: A dictionaty in form of {feature name: (feature start index, feature end index + 1), ... }. This helps with retrieving values from tensor. 
            
        Object methods: 

            self.__init__(): Initialize the object. 
            self.__getitem__(): Returns a feature and a target both as torch tensors, in this order, of a candidate. 
            self.__len__(): Returns the length of the dataset object. 
            get_row_id(): Returns the row_id for the feature and target returned
        """
        super().__init__()
        #If query_str is None: 
        if query_str is None: 
            #First case, no restriction on time and stock id 
            if ((time_id_list is None) & (stock_id_list is None)):
                #Import and pivot time series features 
                if not df_ts_feat is None:
                    df_ts_pv=df_ts_feat.pivot(index="row_id", columns="sub_int_num", values=ts_features).dropna(axis="columns")
                    #Import, add in the target, and pivot tabular features 
                if not df_tab_feat is None: 
                    df_tab_copy=df_tab_feat.copy(deep=True)
                if not df_target is None: 
                    if df_tab_feat is None: 
                        df_tab_copy=df_target
                        tab_features=[]
                    else: 
                        col_diff=list(set(df_target.columns)-set(df_tab_copy.columns))+["row_id"]
                        df_tab_copy=pd.merge(df_tab_copy,df_target[col_diff],on="row_id")
                df_tab_copy["sub_int_num"]=np.nan 
                feat_tar=tab_features+[target]
                df_tab_pv=df_tab_copy.pivot(index="row_id", columns="sub_int_num", values=feat_tar)
                del df_tab_copy 
                #Create the full dataframe 
                df_whole_pv_dna=pd.merge(df_ts_pv,df_tab_pv,on="row_id").dropna(axis="rows")
                del df_ts_pv
                del df_tab_pv
                del feat_tar
            #Second case, only resticting stock id 
            elif (time_id_list is None):
                #Import and pivot time series features 
                if not df_ts_feat is None: 
                    df_ts_pv=df_ts_feat[df_ts_feat["stock_id"].astype(int).isin(stock_id_list)].pivot(index="row_id", columns="sub_int_num", values=ts_features).dropna(axis="columns")
                    #Import, add in the target, and pivot tabular features 
                if not df_tab_feat is None: 
                    df_tab_copy=df_tab_feat[df_tab_feat["stock_id"].astype(int).isin(stock_id_list)]
                if not df_target is None: 
                    if df_tab_feat is None: 
                        df_tab_copy=df_target[df_target["stock_id"].astype(int).isin(stock_id_list)]
                        tab_features=[]
                    else: 
                        col_diff=list(set(df_target.columns)-set(df_tab_copy.columns))+["row_id"]
                        df_tab_copy=pd.merge(df_tab_copy,df_target[df_target["stock_id"].astype(int).isin(stock_id_list)][col_diff],on="row_id")
                df_tab_copy["sub_int_num"]=np.nan 
                feat_tar=tab_features+[target]
                df_tab_pv=df_tab_copy.pivot(index="row_id", columns="sub_int_num", values=feat_tar)
                del df_tab_copy 
                #Create the full dataframe 
                df_whole_pv_dna=pd.merge(df_ts_pv,df_tab_pv,on="row_id").dropna(axis="rows")
                
                del df_ts_pv
                del df_tab_pv
                del feat_tar
            #Thrid case, only restricting time id 
            elif (stock_id_list is None): 
                #Import and pivot time series features 
                if not df_ts_feat is None: 
                    df_ts_pv=df_ts_feat[df_ts_feat["time_id"].astype(int).isin(time_id_list)].pivot(index="row_id", columns="sub_int_num", values=ts_features).dropna(axis="columns")
                    #Import, add in the target, and pivot tabular features 
                
                    #Debug line print
                    # print(df_ts_pv)    
                if not df_tab_feat is None: 
                    #Debug line print 
                    # print(df_tab_feat)
                    
                    df_tab_copy=df_tab_feat[df_tab_feat["time_id"].astype(int).isin(time_id_list)]
                    
                    #Debug line print
                    # print(df_tab_copy) 
                if not df_target is None: 
                    if df_tab_feat is None: 
                        df_tab_copy=df_target[df_target["time_id"].astype(int).isin(time_id_list)]
                        #Debug line print 
                        # print(df_tab_copy)
                        
                        tab_features=[]
                    else: 
                        col_diff=list(set(df_target.columns)-set(df_tab_copy.columns))+["row_id"]
                        df_tab_copy=pd.merge(df_tab_copy,df_target[df_target["time_id"].astype(int).isin(time_id_list)][col_diff],on="row_id")
                        
                        #Debug line print
                        # print(df_tab_copy)
                df_tab_copy["sub_int_num"]=np.nan 
                feat_tar=tab_features+[target]
                df_tab_pv=df_tab_copy.pivot(index="row_id", columns="sub_int_num", values=feat_tar)
                del df_tab_copy 
                #Debug line print 
                # print(df_tab_pv)
            
                #Create the full dataframe 
                df_whole_pv_dna=pd.merge(df_ts_pv,df_tab_pv,on="row_id").dropna(axis="rows")
                del df_ts_pv
                del df_tab_pv
                del feat_tar
                # print(df_whole_pv_dna.columns)
            #Last, and forth, case, restricting both stock and time id
            else: 
                #Import and pivot time series features 
                if not df_ts_feat is None: 
                    df_ts_pv=df_ts_feat[(df_ts_feat["time_id"].astype(int).isin(time_id_list))&(df_ts_feat["stock_id"].astype(int).isin(stock_id_list))].pivot(index="row_id", columns="sub_int_num", values=ts_features).dropna(axis="columns")
                    #Import, add in the target, and pivot tabular features 
                if not df_tab_feat is None: 
                    df_tab_copy=df_tab_feat[(df_tab_feat["time_id"].astype(int).isin(time_id_list))&(df_tab_feat["stock_id"].astype(int).isin(stock_id_list))]
                if not df_target is None:
                    if df_tab_feat is None: 
                        df_tab_copy=df_target[(df_target["time_id"].astype(int).isin(time_id_list))&(df_target["stock_id"].astype(int).isin(stock_id_list))]
                        tab_features=[]
                    else: 
                        col_diff=list(set(df_target.columns)-set(df_tab_copy.columns))+["row_id"]
                        df_tab_copy=pd.merge(df_tab_copy,df_target[(df_target["time_id"].astype(int).isin(time_id_list))&(df_target["stock_id"].astype(int).isin(stock_id_list))][col_diff],on="row_id")
                df_tab_copy["sub_int_num"]=np.nan 
                feat_tar=tab_features+[target]
                df_tab_pv=df_tab_copy.pivot(index="row_id", columns="sub_int_num", values=feat_tar)
                del df_tab_copy 
                #Create the full dataframe 
                df_whole_pv_dna=pd.merge(df_ts_pv,df_tab_pv,on="row_id").dropna(axis="rows")
                del df_ts_pv
                del df_tab_pv
                del feat_tar
        #If query_str is not None, apply query_str filtering and ignoring time_id_list and stock_id_list filtering. 
        else: 
            #Import and pivot time series features 
            if not df_ts_feat is None: 
                df_ts_pv=df_ts_feat.query(query_str).pivot(index="row_id", columns="sub_int_num", values=ts_features).dropna(axis="columns")
                #Import, add in the target, and pivot tabular features 
            if not df_tab_feat is None: 
                df_tab_copy=df_tab_feat.query(query_str)
            if not df_target is None: 
                if df_tab_feat is None: 
                    df_tab_copy=df_target.query(query_str)
                    tab_features=[]
                else: 
                    col_diff=list(set(df_target.columns)-set(df_tab_copy.columns))+["row_id"]
                    df_tab_copy=pd.merge(df_tab_copy,df_target.query(query_str)[col_diff],on="row_id")
            df_tab_copy["sub_int_num"]=np.nan 
            feat_tar=tab_features+[target]
            df_tab_pv=df_tab_copy.pivot(index="row_id", columns="sub_int_num", values=feat_tar)
            del df_tab_copy 
            #Create the full dataframe 
            df_whole_pv_dna=pd.merge(df_ts_pv,df_tab_pv,on="row_id").dropna(axis="rows")
            del df_ts_pv
            del df_tab_pv
            del feat_tar
        #Create object values 
        #The features, targets, and length
        all_feat=ts_features+tab_features

        # Reorders the rows of dataframe numerically (first by 'stock_id', then by 'time_id')
        if numeric==True:
            df_whole_pv_dna = df_whole_pv_dna.loc[
                df_whole_pv_dna.index.to_series()
                .apply(lambda s: tuple(map(int, s.split('-'))))
                .sort_values()
                .index
            ]
        
        # I added this to identify the returned item
        self.row_ids = df_whole_pv_dna.index.to_list()

        
        #Create normalized data 
        if not norm_feature_dict is None: 
            #Create dictionary to save the mean and std
            self.feat_norm_dict=dict()
            for feat in list(norm_feature_dict.keys()): 
                # #Safe the mean and std for recovery purpose 
                # if feat == target: 
                #     self.target_mean=df_whole_pv_dna[feat].values.mean()
                #     self.target_std=df_whole_pv_dna[feat].values.std() 
                #     print("Target mean and std has been recorded.\n")
                #Create the normalized values 
                #Initiate the mean and std values
                feat_mean=None
                feat_std=None 
                #If forced feat mean and std is required  
                if not norm_feature_dict[feat] is None: 
                    feat_mean=norm_feature_dict[feat][0]
                    feat_std=norm_feature_dict[feat][1]
                else: 
                    feat_mean=df_whole_pv_dna[feat].values.mean()
                    feat_std=df_whole_pv_dna[feat].values.std()
                df_whole_pv_dna[feat]=(df_whole_pv_dna[feat]-feat_mean)/feat_std
                #Saving the mean and std
                self.feat_norm_dict[feat]=(feat_mean,feat_std)
                print("Notice: "+feat+" has been normalized.\nThe mean and std of this feature has been stored in feat_norm_dict")
        #Debug line print
        # print(df_whole_pv_dna)
        
        self.features=torch.tensor(df_whole_pv_dna.loc[:,all_feat].values.astype(np.float32),dtype=torch.float32)
        self.target=torch.tensor(df_whole_pv_dna.loc[:,target].values.astype(np.float32),dtype=torch.float32)
        self.len=df_whole_pv_dna.shape[0]
        #The record of feature positions 
        all_feat_len=[df_whole_pv_dna[feat].shape[1] for feat in all_feat]
        self.featuresplit=dict(zip(all_feat,all_feat_len))
        self.featureplace=dict()
        feat_start=0
        for feat in all_feat: 
            #As a reminder, feat_end is one plus the index of the end of the feature 
            feat_end=feat_start+self.featuresplit[feat]
            self.featureplace[feat]=(feat_start,feat_end)
            feat_start=feat_end
        #Clean up
        del df_whole_pv_dna
        del all_feat 
        del all_feat_len
    def __getitem__(self, index):
        # return super().__getitem__(index)
        return self.features[index], self.target[index]
    def __len__(self):
        return self.len
     # I added this to identify the returned item   
    def get_row_id(self, index):
        return self.row_ids[index] 

        
#NNmodel########################################################################################################################################

#Frozen diff creation convolution layer 

class frozen_diff_conv(nn.Module):
    #Created 07/01/25: See Frozen_conv_layer.ipynb for documentation. 
    #Modified 07/03/25. Forcing require_grad = False. 
    def __init__(self,n_diff=1):
        """
        A frozen 1d convolution layer that creates "n th derivative" features for timeseries features. It expects input of tensor shape (N,Channel,Length) with N be any arbitrary positive integer, Channel == 1, and Length be any arbitrary integer. 
        
        :param n_diff: Defaulted to 1. The number of derivative wanted. 
        :return: A tensor of shape (N, n_diff, Length). Where the n th (start from zero) tensor in the dimension 1 (we start with dimension 0) is the n th "derivative" of the imput tensor. However, if n_diff >= Length, None will be returned. 
        """
        super().__init__()
        self.n_diff=n_diff
        self.frozen_conv=nn.Conv1d(1,1,kernel_size=2,bias=False)
        with torch.no_grad():
            self.frozen_conv.weight[:]=torch.tensor([[[-1.0,1.0]]])
            # self.frozen_conv.bias.zero_()
        for param in self.frozen_conv.parameters():
            param.requires_grad = False

    def forward(self,x):
        out_tensor=x
        x_diff=x
        #Check if the user is taking too many derivatives 
        if self.n_diff>=x.shape[2]: 
            print("Too many derivatives. Returning None.\n")
            return None
        for diff in range(1,self.n_diff+1): 
            x_diff=self.frozen_conv(x_diff)
            x_diff_pad=F.pad(x_diff,(0,diff),mode="constant",value=0)
            out_tensor=torch.cat((out_tensor,x_diff_pad),dim=1)
            
        # x_diff=self.frozen_conv(x)
        # x_diff_pad=F.pad(x_diff,(0,1),mode="constant",value=0)
        # out_tensor=torch.cat((x,x_diff_pad),dim=1)
        
        return out_tensor
    
#Basic rnn layer

class RV_RNN_conv(nn.Module):        
    #Created 07/02/25 see RNN_with_frozen_conv.ipynb for documentation. 
    #Modified 07/08/25 Added LSTM and GRU options
    def __init__(self,rnn_num_layer,rnn_drop_out,n_diff = 2,rnn_type="rnn",rnn_act="tanh",proj_dim=32,rnn_hidden_size=32,input_scaler=10000):
        """
        :param n_diff: Defaulted to 2. Decides how many derivative features is wanted in the time series. 
        :param rnn_num_layer: num_layer parameter for rnn. 
        :param rnn_drop_out: dropout parameter for rnn. 
        :param rnn_act: Defaulted to "tanh". Nonlinearity parameter for rnn. 
        :param proj_dim: Defaulted to 32. Decided the dimension of projection before feeding into rnn. 
        :param rnn_hidden_size: Defaulted to 32. The hidden_size parameter for rnn. 
        :param input_scaler: Defaulted to 10000. Set a scaling to input, a lot of timeseries values of our data are extremely close to zero. 
        :param rnn_type: 'rnn', 'lstm', or 'gru'
        """
        super().__init__()
        
        self.input_scaler=input_scaler
        self.frozen_conv=frozen_diff_conv(n_diff=n_diff)
        self.linear_proj_input=nn.Linear(n_diff+1,proj_dim)

        self.rnn_type = rnn_type

        if rnn_type == "rnn":
            self.RNN_layer=nn.RNN(input_size=proj_dim,
                                  hidden_size=rnn_hidden_size,
                                  num_layers=rnn_num_layer,
                                  nonlinearity=rnn_act,
                                  batch_first=True,
                                  dropout=rnn_drop_out)
        elif rnn_type == "lstm":
            if rnn_act is not None:
                print(f"Warning: rnn_act='{rnn_act}' is ignored when using rnn_type='lstm'")
            self.RNN_layer = nn.LSTM(input_size=proj_dim,
                                     hidden_size=rnn_hidden_size,
                                     num_layers=rnn_num_layer,
                                     batch_first=True,
                                     dropout=rnn_drop_out)
        elif rnn_type == "gru":
            if rnn_act is not None:
                print(f"Warning: rnn_act='{rnn_act}' is ignored when using rnn_type='gru'")
            self.RNN_layer = nn.GRU(input_size=proj_dim,
                                    hidden_size=rnn_hidden_size,
                                    num_layers=rnn_num_layer,
                                    batch_first=True,
                                    dropout=rnn_drop_out)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")
        
        self.linear_post_rnn=nn.Linear(rnn_hidden_size,1)
        self.frozen_list=["frozen_conv"] 
        
    def forward(self,x):
        #First, scale the input, and unsqueese to add in one dimension in dim 1 as channel. This is needed for convolution. 
        x*=self.input_scaler
        x=torch.unsqueeze(x,dim=1)
        x=self.frozen_conv(x)
        x=x.permute(0,2,1)
        x=self.linear_proj_input(x)
        x=self.RNN_layer(x)[0]
        x=self.linear_post_rnn(x)
        
        return torch.sum(x,dim=1)/self.input_scaler

# id adjusted rnn model by multiplication 
    
class id_learned_embedding_adj_rnn_mtpl(nn.Module): 
    #Created 07/21/25
    def __init__(self, ts_place, id_place, rnn_model, id_hidden_model, id_input_num=112,emb_dim=8):
        """
        A model that takes a categorical id and embed it to a higher dimensional vecotr space, then use the embedded vector as for adjustment on the base rnn models. 
        
        :param ts_place: A tuple indicating the place of timeseries input in the full input tensor. 
        :param id_palce: A tuple indicating the place of category id input in the full input tensor. 
        :param rnn_model: The base rnn_model to be adjusted. 
        :param id_hidden_model: The hidden layers applied to the embedded (into higher dimensional vector space) categorical id. 
        :param id_input_num: Defaulted to 112, the total number of stocks in our project. In general, this should be the total level of the concerned catregorical id. 
        :param emb_dim: Defaulted to 8. The desired dimension of the vector space that one wants to embed the categorical id into. 
        """
        super().__init__()
        self.id_embeder = nn.Embedding(num_embeddings=id_input_num, embedding_dim=emb_dim)
        self.rnn_layer = rnn_model 
        self.id_hidden_model = id_hidden_model
        
        self.ts_place=ts_place
        self.id_place=id_place
    def forward(self,x):
        x_ts, emb_id = x[:,self.ts_place[0]:self.ts_place[1]], x[:,self.id_place[0]:self.id_place[1]].long() 
        
        rnn_output=self.rnn_layer(x_ts)
        adj_value=self.id_hidden_model(self.id_embeder(emb_id)).squeeze(2)
        
        #Print shape for debugging 
        # print("rnn shape", rnn_output.shape)
        # print("adj shape", adj_value.shape)
        
        return rnn_output*adj_value
    
# id adjusted rnn model with attention 

class id_learned_embedding_attend_rnn(nn.Module): 
    #Created 07/23/25
    def __init__(self, ts_place, id_place, rnn_model, id_hidden_model, id_input_num=112, id_emb_dim=8, att_emb_dim=32,att_num_head=1): 
        """
        A model that takes rnn output and simply have it pay (cross) attention to the categorical id. 
        
        :param ts_place: A tuple indicating the place of timeseries input in the full input tensor. 
        :param id_palce: A tuple indicating the place of category id input in the full input tensor. 
        :param rnn_model: The base rnn_model to be adjusted. 
        :param id_hidden_model: The hidden layers applied to the embedded (into higher dimensional vector space) categorical id. 
        :param id_input_num: Defaulted to 112, the total number of stocks in our project. In general, this should be the total level of the concerned catregorical id. 
        :param id_emb_dim: Defaulted to 8. The desired dimension of the vector space that one wants to embed the categorical id into. 
        :param att_emb_dim: Defaulted to 32. The emb_dim used by the cross attention layer. 
        :param att_num_head: Defaulted to 1. The num_heads used by the cross attention layer. 
        """
        super().__init__()
        self.id_embeder = nn.Embedding(num_embeddings=id_input_num, embedding_dim=id_emb_dim)
        self.rnn_layer = rnn_model 
        self.id_hidden_model = id_hidden_model
        self.attention=nn.MultiheadAttention(embed_dim=att_emb_dim,num_heads=att_num_head,dropout=0.1,batch_first=True)
        self.post_rnn_linear=nn.Linear(in_features=1,out_features=32)
        self.final_linear=nn.Linear(in_features=att_emb_dim,out_features=1)
        
        self.ts_place=ts_place
        self.id_place=id_place
    def forward(self,x): 
        x_ts, emb_id = x[:,self.ts_place[0]:self.ts_place[1]], x[:,self.id_place[0]:self.id_place[1]].long() 
        
        rnn_output=self.post_rnn_linear(self.rnn_layer(x_ts).unsqueeze(2))
        adj_output=self.id_hidden_model(self.id_embeder(emb_id))
        #debug print dim
        # print("rnn and adj ready")
        # print("rnn", rnn_output.shape)
        # print("adj", adj_output.shape)
        att_output,_=self.attention(query=adj_output,key=rnn_output,value=rnn_output) 
        # print("att ready")
        output=self.final_linear(att_output)
        
        return torch.sum(output,dim=1)
    
# Positional embedding by cross attention 

class pos_emb_cross_attn(nn.Module): 
    #Created 07/24/25 
    def __init__(self,length,ts_dim,emb_dim,dropout,num_heads): 
        """
        Takes time series x of shape (Batch size, length, ts_dim), and produces layernorm(x+ cross_attn(q=x,k=position,v=postion)) that has dimension emb_dim in each time step. 
        
        :param length: The length of each timeseries. 
        :param ts_dim: The dimension of each time step. 
        :param emb_dim: The dimension to which one wants to project each time step. 
        :param dropout: The dropout rate used by the cross attention layer. 
        :param num_heads: The num_heads used by cross attention layer. 
        :return: layernorm(x+ cross_attn(q=x,k=position,v=postion)). 
        """
        super().__init__()
        self.length=length 
        self.ts_proj=nn.Linear(in_features=ts_dim,out_features=emb_dim)
        self.pos_emb=nn.Embedding(num_embeddings=length,embedding_dim=emb_dim) # 60 is the length of our (default) timeseries. 
        self.pos_attn=nn.MultiheadAttention(embed_dim=emb_dim,batch_first=True,dropout=dropout,num_heads=num_heads)
        self.pos_norm=nn.LayerNorm(emb_dim) 
        
    def forward(self,x):
        batch_num=x.shape[0]
        x=self.ts_proj(x)
        pos_id=torch.arange(self.length).expand(batch_num,60).to(device=x.device)
        pos_emb=self.pos_emb(pos_id)
        pos,_=self.pos_attn(x,pos_emb,pos_emb)
        x=x+pos
        x=self.pos_norm(x)
        
        return x
        
# Encoder for timeseries 

class ts_encoder(nn.Module): 
    #Created 07/24/25 
    def __init__(self,ts_dim,dropout,num_heads,feedforward_layer_list): 
        """
        An encoder (self attention) layer designed for timeseries. Takes timeseries of shape (batch size, length, ts_dim). 
        
        :param ts_dim: The dimention of each time step. 
        :param dropout: The drop out rate of the self attention layer. 
        :param num_heads: The num_heads used by the self attention layer. 
        :param feedforward_layers_list: The list of feed forward layer post the self attention layer. Must take tensor in shape of (batch size, length, ts_dim). For our purpose, it is also advices to have it output the same shape. 
        :return: A tensor of shape (batch size, length, step dimension of feedforward_layers)
        """
        super().__init__()
        self.encoder_attn=nn.MultiheadAttention(embed_dim=ts_dim,num_heads=num_heads,dropout=dropout,batch_first=True)
        self.encoder_norm1=nn.LayerNorm(ts_dim)
        # ff_list=[]
        # current_dim=ts_dim
        # for i in range(1,feedforward_layer_num+1): 
        #     if i==feedforward_layer_num: 
        #         ff_list.append(nn.Linear(in_features=feedforward_dim,out_features=ts_dim))
        #     elif i==1: 
        #         ff_list.append(nn.Linear(in_features=ts_dim,out_features=feedforward_dim))
        #         ff_list.append(nn.ReLU())
        #     else: 
        #         ff_list.append(nn.Linear(in_features=feedforward_dim,out_features=feedforward_dim))
        #     # current
        self.encoder_feedforward=nn.ModuleList(feedforward_layer_list)
        self.encoder_norm2=nn.LayerNorm(ts_dim)
        
    def forward(self,x):
        attn,_=self.encoder_attn(x,x,x)
        x=self.encoder_norm1(x+attn)
        attn=x
        for layer in self.encoder_feedforward: 
            attn=layer(attn)
        # attn=self.encoder_feedforward(x)
        
        return self.encoder_norm2(x+attn)
        
# Encoder ensemble 

class encoder_ensemble(nn.Module): 
    #Created 07/24/25 
    def __init__(self,pos_emb_model,output_feedforward,encoder_dropout,encoder_feedforward_list,n_diff=2,encoder_layer_num=4,input_scaler=10000,ts_emb_dim=32,encoder_num_heads=4): 
        """
        The ensemble of compoenents to produce a whole transformer (encoder based only) model for timeseries to predict the target. 
        
        :param pos_emb_model: The postion embedding model. Expected to be created through pos_emb_cross_attn(). 
        :param output_feedforward: The feed forward layers right before the output. Expected to take tensors of out put shape of the encoder_model and produce a tensor of shape (batch size, length, 1). 
        :param n_diff: Defaulted to 2. The number of dirivatives to be created for the timeseries. 
        :param encoder_layer_num: Defaulted to 4. The number of layers of encoder to be applied in a row. 
        :param input_scaler: Defaulted to 10000: The input scaler to make the input values more reasonably sized. 
        :param encoder_dropout: The dropout rate of encoder. 
        :param encoder_feedforward_list: The feedforward_layer_list used by the encoder. 
        :param encoder_num_heads: The num_heads used by the encoder. 
        :param ts_emb_dim: Defaulted to 32. The dimension of each time step of the post pos_emb_model timeseries. 
        :param device: The device used. 
        """
        super().__init__() 
        #Frozen convolution 
        # self.ts_proj=nn.Linear(in_features=n_diff+1,out_features=ts_emb_dim)
        self.frozen_conv=frozen_diff_conv(n_diff=n_diff)
        #Position embedding 
        self.pos_emb=pos_emb_model
        #Encoder layers
        self.encoder_layers=nn.ModuleList([
            ts_encoder(ts_dim=ts_emb_dim,num_heads=encoder_num_heads,dropout=encoder_dropout,feedforward_layer_list=encoder_feedforward_list)
            for _ in range(encoder_layer_num)
        ])
        #Output feedforward
        self.output_feedforward=output_feedforward 
        
        #Scaler 
        self.input_scaler=input_scaler
        
    def forward(self,x):
        #Adjust input 
        x*=self.input_scaler
        x=torch.unsqueeze(x,dim=1)
        x=self.frozen_conv(x)
        x=x.permute(0,2,1) 
        #Position embedding 
        x=self.pos_emb(x)
        #Run though the encoder layers 
        for layer in self.encoder_layers: 
            x=layer(x)
        #Output feedforward
        x=self.output_feedforward(x)
        return torch.sum(x,dim=1)/self.input_scaler