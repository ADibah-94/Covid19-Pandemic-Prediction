# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:02:33 2022

@author: HP
"""
    
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


class ExploratoryDataAnalysis():
    
    def __init(self):
        pass
    
    def remove_tags(self,data):
        '''


        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        '''
        for i in range(window_size,len(train_df_new)):  # window_size, max number of row
            X_train.append(train_scaled[i-window_size:i,0])
            Y_train.append(train_scaled[i,0])
            
        return data 
    
    def lower_split(self,data):
        '''
        This function converts all letters into lowercase and split into list.
        Also filters numerical data
        
        Parameters
        ----------
        data : Array
            RAW TRAINING DATA CONTAINING STRINGS.

        Returns
        -------
        data : List
            CLEANED DATA WITH ALL LETTERS CONVERTED INTO LOWERCASE.

        '''
        
        for i in range(window_size,len(temp)):
            X_test.append(temp[i-window_size:i,0])
            Y_test.append(temp[i,0])
            
        return data
        
class ModelCreation():
    
    def __init__(self):
        pass
    
    
    def lstm_layer(self, nb_categories, nodes=64, dropout=0.2):
        
        model = Sequential()
        model.add(LSTM(nodes,activation='tanh',
               return_sequences=(True), 
               input_shape=(X_train.shape[1],1))) 
        model.add(Dropout(dropout))
        model.add(LSTM(nodes))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories))
        model.summary()
        
        return model
    
   
#%% 
if __name__ == '__main__':
    
    import os
    import pandas as pd

    TRAIN_DATASET = os.path.join(os.getcwd(),'Datasets',
                             'cases_malaysia_train.csv')
    TEST_DATASET = os.path.join(os.getcwd(), 'Datasets',
                            'cases_malaysia_test.csv')
    MMS_SCALER_SAVE_PATH = os.path.join(os.getcwd(),'saved_model',
                                       'mms_scaler.pkl')
    LOG_PATH = os.path.join(os.getcwd(), 'logs')
    MODEL_PATH = os.path.join(os.getcwd(), 'saved_model','model.h5') 

    train_df = pd.read_csv(TRAIN_DATASET)
    test_df = pd.read_csv(TEST_DATASET)
    

    #%%

    eda = ExploratoryDataAnalysis()





    





