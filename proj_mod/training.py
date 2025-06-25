import numpy as np
import torch
import torch.nn as nn
import sys
from sklearn.linear_model import LinearRegression


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
    RMSPE is not a premade pytorch loss function, the following creates the loss function so that it can be used in training. 
    """
    def __init__(self,eps=sys.float_info.epsilon): 
        super(RMSPELoss,self).__init__()
        self.eps=eps
        
    def forward(self, ypred, ytrue): 
        return torch.sqrt(torch.mean(torch.square((ypred-ytrue)/(ytrue+self.eps))))
        