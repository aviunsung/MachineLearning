#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 16:34:20 2018

@author: Avinash
"""

import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns                            # more plots
from fbprophet import Prophet

#Mean Absolute Percentage Error (MAPE)
def mape(y_pred,y_true):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_accuracy(actual,predicted, perError):
    if len(actual) != len(predicted):
        return -1
    pr = (np.abs((actual.values - predicted.values))/ actual)*100
    pr = pr <= perError
    perc = (np.sum(pr)/len(pr))*100
    return perc

#Calculate RMSE and MAPE
def calculateError(input_data,predicted_data):
    rmse = sqrt(mean_squared_error(input_data, predicted_data))
    map_error=mape(input_data, predicted_data)
    accuracy=calculate_accuracy(input_data, predicted_data,10)
    print('RMS='+str(rmse))
    print('MAPE='+str(map_error))
    print('Accuracy='+str(accuracy))
 
train_data=pd.read_excel('/home/avinash/Desktop/ML_Assignments/InventoryForecasting/forecasting_train_dataset.xlsx')
daily_train=pd.DataFrame()

daily_train['ds'] = train_data['date']
daily_train['y'] = train_data['totalRequests']

test_data = pd.read_excel('/home/avinash/Desktop/ML_Assignments/InventoryForecasting/forecasting_test_dataset.xlsx')

#Initialize Prophet

m = Prophet(weekly_seasonality=True)
#m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.fit(daily_train)
future = m.make_future_dataframe(periods=130)
forecast = m.predict(future)
fig = m.plot_components(forecast)

# =============================================================================
# m = Prophet()
# m.fit(daily_train)
# 
# future = m.make_future_dataframe(periods=130)
# print(future.tail())
# 
# forecast = m.predict(future)
# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
# 
# =============================================================================
#fig1 = m.plot(forecast)

calculateError(test_data.totalRequests,forecast.yhat[410:])




