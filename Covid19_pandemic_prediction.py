# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:09:26 2022

@author: HP
"""

import pandas as pd
import os 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import datetime 
from  sklearn.metrics import mean_absolute_error

#%% Paths 

TRAIN_DATASET = os.path.join(os.getcwd(),'Datasets',
                             'cases_malaysia_train.csv')
TEST_DATASET = os.path.join(os.getcwd(), 'Datasets',
                            'cases_malaysia_test.csv')
MMS_SCALER_SAVE_PATH = os.path.join(os.getcwd(),'saved_model','mms_scaler.pkl')
LOG_PATH = os.path.join(os.getcwd(), 'logs')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_model','model.h5') 
#%% EDA 

# Step 1) Data loading

train_df = pd.read_csv(TRAIN_DATASET)
test_df = pd.read_csv(TEST_DATASET)

# Step 2) Data Interpretation

train_df.describe()
train_df.info()           # To view the summary of dataframe
train_df.isnull().sum()   # to check any missing values

test_df.info()
test_df.describe()
test_df.isnull().sum() 

# Step 3) Data cleaning

# For training dataset

train_df = train_df.drop(columns='date')    # drop date

# Use label encoder to convert object to numeric
label_enc = LabelEncoder()
train_df['cases_new'] = label_enc.fit_transform(train_df['cases_new'])

# Filling NaN using Iterative Imputer
imputer = IterativeImputer()
train_df_new = imputer.fit_transform(train_df)

# For testing dataset

test_df = test_df.drop(columns='date')    # drop date

# Filling NaN using Iterative Imputer
imputer = IterativeImputer()
test_df_new = imputer.fit_transform(test_df)

# Step 4) Data preprocessing

# Perform min-max scaling 

scaler = MinMaxScaler()  
train_scaled = scaler.fit_transform(train_df_new)
pickle.dump(scaler, open(MMS_SCALER_SAVE_PATH,'wb'))

test_scaled = scaler.fit_transform(test_df_new)

window_size = 30   # 30 days is used to predict what next value

# Training Dataset

X_train=[]
Y_train=[]

for i in range(window_size,len(train_df_new)):  # window_size, max number of row
    X_train.append(train_scaled[i-window_size:i,0])
    Y_train.append(train_scaled[i,0])
     
# Convert to array
X_train=np.array(X_train)
Y_train=np.array(Y_train)

# Testing Dataset
# test_scaled, train_scaled # Both in array
temp = np.concatenate((train_scaled,test_scaled))
length_window = window_size+len(test_scaled)
temp = temp[-length_window:]

X_test=[]
Y_test=[]

for i in range(window_size,len(temp)):
    X_test.append(temp[i-window_size:i,0])
    Y_test.append(temp[i,0])
    
X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

#%% Model Creation

model = Sequential()
model.add(LSTM(64,activation='tanh',
               return_sequences=(True), 
               input_shape=(X_train.shape[1],1))) 
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()

#%% Calbacks 

log_dir = os.path.join(LOG_PATH, 
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

# Tensorboard callback

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Early stopping callback

early_stopping_callback= EarlyStopping(monitor='loss', patience=3)

#%% Compile ad Model Fitting

model.compile(optimizer='adam',
              loss='mse',
              metrics='mse')

hist = model.fit(X_train,Y_train, 
                 epochs=50, batch_size=128,
                 callbacks=[tensorboard_callback,early_stopping_callback])

print(hist.history.keys())

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['mse'])
plt.show()

#%% Model Evaluation

predicted = []    # (batch, length, features)
              
for test in X_test:
    predicted.append(model.predict(np.expand_dims(test, axis=0)))

predicted = np.array(predicted)

#%% Model Analysis

plt.figure()
plt.plot(predicted.reshape(len(predicted),1)) 
plt.plot(Y_test)
plt.legend(['Predicted', 'Actual'])
plt.show()

y_true = Y_test
y_pred = predicted.reshape(len(predicted),1)

print((mean_absolute_error(y_true, y_pred)/sum(abs(y_true))) *100)

# So mean absolute percentage error less than 1% which is 0.5525

#%% Model Deployment

model.save(MODEL_PATH)

