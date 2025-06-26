import numpy as np
import torch
import torch.nn as nn
import sys
import time
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
     
def reg_validator_rmspe(model, val_loader, eps=sys.float_info.epsilon, device): 
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
        
   
def reg_training_loop_rmspe(n_epochs=1000, optimizer, model, train_loader, val_loader, ot_steps=100, recall_best=True, device, eps=sys.float_info.epsilon, list_train_loss=None, list_val_loss=None, report_interval=20): 
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