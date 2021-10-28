#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Market prediction for next month using Reccurent Neural Network

@author: Deronsart
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt



if __name__ == '__main__':
    
    # Variables used
    
    training_set_file = 'tests/Stock_Price_Train_5_years.csv'
    test_set_file = 'tests/Stock_Price_Test_5_years.csv'
    number_month = 3                                                           # number of months observed to predict the output
    
    batch = 32
    nb_epochs = 100
    
    
    
    # I/ Data Preprocessing
    
    dataset_train = pd.read_csv(training_set_file)
    training_set = dataset_train.iloc[:, 1:2].values
    
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set = sc.fit_transform(training_set)
    
    X_train = []
    y_train = []
    timesteps = number_month * 20                                              # 1 stock market month = 20 days
    for i in range(timesteps, len(training_set)):
        X_train.append(training_set[i - timesteps : i, 0])
        y_train.append(training_set[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
        
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    
    
    # II/ Building the RNN
    
    regressor = Sequential()
    
    regressor.add(LSTM(units=50, activation='tanh', recurrent_activation='sigmoid', 
                       return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    
    regressor.add(LSTM(units=75, return_sequences=True))
    regressor.add(Dropout(0.2))
    
    regressor.add(LSTM(units=75, return_sequences=True))
    regressor.add(Dropout(0.2))
    
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
        
    regressor.add(Dense(units=1))
        
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.fit(X_train, y_train, batch_size=batch, epochs=nb_epochs)
    
    
    
    # Making the predictions
        
    dataset_test = pd.read_csv(test_set_file)
    y_test = dataset_test.iloc[:, 1:2].values
        
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
        
    inputs = dataset_total[len(dataset_total)-len(dataset_test)-timesteps:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
        
    X_test = []
    for i in range(timesteps, timesteps + 20):
        X_test.append(inputs[i - timesteps : i, 0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    y_pred = regressor.predict(X_test)
    y_pred = sc.inverse_transform(y_pred)
    
    
    
    # Visualising the results
    
    plt.plot(y_test, color='red', label='Real Stock Price')
    plt.plot(y_pred, color='blue', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Stock Market Days')
    plt.xticks(np.arange(0, 21, 1))
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


