import numpy as np
import torch
import torch.nn as nn
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
    
    :param eps: Defaulted to sys.float_info.epsilon. The small value needed to avoid division by zero. 
    """
    def __init__(self,eps=sys.float_info.epsilon): 
        super(RMSPELoss,self).__init__()
        self.eps=eps
        
    def forward(self, ypred, ytrue): 
        return torch.sqrt(torch.mean(torch.square((ypred-ytrue)/(ytrue+self.eps))))
    
class MSPELoss(nn.Module): 
    #Created 06/25/25 
    """
    Mean squared percentage error is not a premade pytorch loss function, the following creates the loss function so that it can be used in training, if needed. 
    
    :param eps: Defaulted to sys.float_info.epsilon. The small value needed to avoid division by zero. 
    """
    def __init__(self,eps=sys.float_info.epsilon): 
        super(RMSPELoss,self).__init__()
        self.eps=eps
        
    def forward(self, ypred, ytrue): 
        return torch.mean(torch.square((ypred-ytrue)/(ytrue+self.eps)))
    
#Training loop###################################################################################################################################
     
def reg_validator_rmspe(model, val_loader, device, eps=sys.float_info.epsilon): 
    #Created 06/25/25 In progress, testing needed. 
    """
    Returns the rmspe on the validation set for regression type training. 
    
    :param model: The model used. 
    :param val_loader: The loader that feeds the validation set. 
    :param eps: Defaulted to sys.float_info.epsilon. The small value needed to avoid division by zero. 
    :param device: The device used to calculate. 
    :return: The rmspe on the validation set. 
    """
    sum_of_square=0
    total_count=len(val_loader.dataset)
    if total_count==0: 
        print("There is nothing in the validation set, returning None")
        return None
    with torch.no_grad():
        for feature, target in val_loader: 
            feature=feature.to(device=device)
            target=target.to(device=device)
            pred=model(feature)
            sum_of_square+=torch.square((pred-target)/(target+eps))
        rmspe=torch.sqrt(sum_of_square/total_count)
    return rmspe
        
def reg_training_loop_rmspe(optimizer, model, train_loader, val_loader, device, ot_steps=100, recall_best=True, eps=sys.float_info.epsilon, list_train_loss=None, list_val_loss=None, report_interval=20, n_epochs=1000): 
    #Created 06/25/25 In progress, testing needed
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
    :param eps: Defaulted to sys.float_info.epsilon. The small value needed to avoid division by zero. 
    :param list_train_loss: Defaulted to None. If set to certain list, the function will append the training loss values to the end of the list in order of epochs. 
    :param list_val_loss: Defaulted to None. If set to certain list, the function will append the validation loss values to the end of the list in order of epochs. 
    :param report_interval: Defaulted to 20. The training loop will report once every report interval number of epochs. 
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
            #First, the standard training steps 
            feature=feature.to(device=device)
            target=target.to(device=device)
            pred=model(feature)
            loss_step=RMSPELoss(pred,target)
            #Update using optimizer according to loss of the batch
            optimizer.zero_grad()
            loss_step.backward()
            optimizer.step()            
            #Update the sum of sqaure (without grad) 
            with torch.no_grad():
                sum_of_sqaure_train+=torch.square((pred-target)/(target+eps))
        #End of training loop for a batch######
        #######################################
        curr_time=time.time()
        time_cost=curr_time-start_time
        #Calculate the training loss of this epoch (without grad)
        with torch.no_grad():    
            epoch_train_loss=torch.sqrt(sum_of_sqaure_train/total_data_count)
        #Calculate the validation loss of this epoch (without grad, citing another function)
        epoch_val_loss=reg_validator_rmspe(model=model,val_loader=val_loader,eps=eps)
        #Update the best validation loss and the epoch that it occurred     
        if ((epoch==1) or (epoch_val_loss<best_val_loss)): 
                best_val_loss=epoch_val_loss
                best_val_epoch=epoch
                #Remember the best model (based on validation loss), store the state dictionary 
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
    def __init__(self, query_str=None, query_val_list=None, time_id_list=None, stock_id_list=None, tab_features=None, ts_features=None, target="target", df_ts_feat=None, df_tab_feat=None, df_target=None):
        """
        Object in subclass of Dataset. 
        
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
        
        Object attributes: 
        
            self.features: The collection of features as a torch tensor. 
            self.target: The collection of targets as a torch tensor. 
            self.len: The length of the whole dataset object. 
            self.featuresplit: A dictionary in form of {feature name:length of feature, ...} to help distinguish different features in the feature torch tensor. The length of feature is included since some of the features are time series, while some are tabular. 
            
        Object methods: 

            self.__init__(): Initialize the object. 
            self.__getitem__(): Returns a feature and a target both as torch tensors, in this order, of a candidate. 
            self.__len__(): Returns the length of the dataset object. 
        """
        super().__init__()
        #If query_str is None: 
        if query_str is None: 
            #First case, no restriction on time and stock id 
            if ((time_id_list==None) & (stock_id_list==None)):
                #Import and pivot time series features 
                df_ts_pv=df_ts_feat.pivot(index="row_id", columns="sub_int_num", values=ts_features).dropna(axis="columns")
                #Import, add in the target, and pivot tabular features 
                df_tab_copy=df_tab_feat.copy(deep=True)
                if not df_target is None: 
                    df_tab_copy=pd.merge(df_tab_copy,df_target,on="row_id")
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
            elif (time_id_list==None):
                #Import and pivot time series features 
                df_ts_pv=df_ts_feat[df_ts_feat["stock_id"].isin(stock_id_list)].pivot(index="row_id", columns="sub_int_num", values=ts_features).dropna(axis="columns")
                #Import, add in the target, and pivot tabular features 
                df_tab_copy=df_tab_feat[df_tab_feat["stock_id"].isin(stock_id_list)]
                if not df_target is None: 
                    df_tab_copy=pd.merge(df_tab_copy,df_target[df_target["stock_id"].isin(stock_id_list)],on="row_id")
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
            elif (stock_id_list==None): 
                #Import and pivot time series features 
                df_ts_pv=df_ts_feat[df_ts_feat["time_id"].isin(time_id_list)].pivot(index="row_id", columns="sub_int_num", values=ts_features).dropna(axis="columns")
                #Import, add in the target, and pivot tabular features 
                df_tab_copy=df_tab_feat[df_tab_feat["time_id"].isin(time_id_list)]
                if not df_target is None: 
                    df_tab_copy=pd.merge(df_tab_copy,df_target[df_target["time_id"].isin(time_id_list)],on="row_id")
                df_tab_copy["sub_int_num"]=np.nan 
                feat_tar=tab_features+[target]
                df_tab_pv=df_tab_copy.pivot(index="row_id", columns="sub_int_num", values=feat_tar)
                del df_tab_copy 
                #Create the full dataframe 
                df_whole_pv_dna=pd.merge(df_ts_pv,df_tab_pv,on="row_id").dropna(axis="rows")
                del df_ts_pv
                del df_tab_pv
                del feat_tar
                # print(df_whole_pv_dna.columns)
            #Last, and forth, case, restricting both stock and time id
            else: 
                #Import and pivot time series features 
                df_ts_pv=df_ts_feat[(df_ts_feat["time_id"].isin(time_id_list))&(df_ts_feat["stock_id"].isin(stock_id_list))].pivot(index="row_id", columns="sub_int_num", values=ts_features).dropna(axis="columns")
                #Import, add in the target, and pivot tabular features 
                df_tab_copy=df_tab_feat[(df_tab_feat["time_id"].isin(time_id_list))&(df_tab_feat["stock_id"].isin(stock_id_list))]
                if not df_target is None:
                    df_tab_copy=pd.merge(df_tab_copy,df_target[(df_target["time_id"].isin(time_id_list))&(df_target["stock_id"].isin(stock_id_list))],on="row_id")
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
            df_ts_pv=df_ts_feat.query(query_str).pivot(index="row_id", columns="sub_int_num", values=ts_features).dropna(axis="columns")
            #Import, add in the target, and pivot tabular features 
            df_tab_copy=df_tab_feat.query(query_str)
            if not df_target is None: 
                df_tab_copy=pd.merge(df_tab_copy,df_target.query(query_str),on="row_id")
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
        self.features=torch.tensor(df_whole_pv_dna.loc[:,all_feat].values.astype(np.float32),dtype=torch.float32)
        self.target=torch.tensor(df_whole_pv_dna.loc[:,target].values.astype(np.float32),dtype=torch.float32)
        self.len=df_whole_pv_dna.shape[0]
        #The record of feature positions 
        all_feat_len=[df_whole_pv_dna[feat].shape[1] for feat in all_feat]
        self.featuresplit=dict(zip(all_feat,all_feat_len))
        #Clean up
        del df_whole_pv_dna
        del all_feat 
        del all_feat_len
    def __getitem__(self, index):
        # return super().__getitem__(index)
        return self.features[index], self.target[index]
    def __len__(self):
        return self.len
        
#NNmodel########################################################################################################################################