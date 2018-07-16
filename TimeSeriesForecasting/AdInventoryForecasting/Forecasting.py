#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 14:32:06 2018

@author: Avinash Pandey
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

evaluation_metrics=pd.DataFrame()

#Mean Absolute Percentage Error (MAPE)
def mape(y_pred,y_true):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#Calculate RMSE and MAPE
def calculateError(input_data,predicted_data,model_name):
    rmse = sqrt(mean_squared_error(input_data, predicted_data))
    map_error=mape(input_data, predicted_data)
    print('RMSE='+str(rmse))
    print('MAPE='+str(map_error))
    evaluation_metrics.set_value(model_name,'RMSE',rmse)
    evaluation_metrics.set_value(model_name,'MAPE',map_error)

def plotGraph(predicted_dataframe,column_name,method_label):
    plt.figure(figsize=(16,8))
    plt.plot(train_data.index, train_data['totalRequests'], label='Train')
    plt.plot(test_data.index,test_data['totalRequests'], label='Test')
    plt.plot(predicted_dataframe.index,predicted_dataframe[column_name], label=method_label)
    plt.legend(loc='best')
    plt.title(method_label)
    plt.show()

def test_stationarity(timeseries_data):
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries_data, window=12)
    rolstd = pd.rolling_std(timeseries_data, window=12)

    #Plot rolling statistics:
    plt.plot(timeseries_data, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
train_data=pd.read_excel("/home/avinash/Learning/MachineLearning/TimeSeriesForecasting/AdInventoryForecasting/Dataset/forecasting_train_dataset.xlsx")
#train_data.index=train_data.date

test_data = pd.read_excel('/home/avinash/Learning/MachineLearning/TimeSeriesForecasting/AdInventoryForecasting/Dataset/forecasting_test_dataset.xlsx')

#Visualze Dataset
#Total Requests
plt.figure(figsize=(16,8))
plt.plot(train_data.date, train_data['totalRequests'], label='Train_totalRequests')
plt.plot(test_data.date,test_data['totalRequests'], label='Test_totalRequests')
plt.legend(loc='best')
plt.title('Total Ad Requests Data')

#Paid Impressions
plt.figure(figsize=(16,8))
plt.plot(train_data.date, train_data['paidImpressions'], label='Train_paidImpressions')
plt.plot(test_data.date,test_data['paidImpressions'], label='Test_paidImpressions')
plt.legend(loc='best')
plt.title('Total Paid Impressions Data')

#1. Naive Method
dd= np.asarray(train_data.totalRequests)
y_hat = test_data.copy()
y_hat['naive'] = dd[len(dd)-1]
plotGraph(y_hat,'naive','Naive Forecast')
calculateError(test_data.totalRequests,y_hat.naive,'Naive')


#2. Simple Avergae
y_hat_avg = test_data.copy()
y_hat_avg['avg_forecast'] = train_data['totalRequests'].mean()
plotGraph(y_hat_avg,'avg_forecast','Average Forecast')
calculateError(test_data.totalRequests,y_hat_avg.avg_forecast,'Simple Average')

#3. Moving Window/Sliding technique
y_hat_mvng_avg = test_data.copy()
y_hat_mvng_avg['moving_avg_forecast'] = train_data['totalRequests'].rolling(120).mean().iloc[-1]
plotGraph(y_hat_mvng_avg,'moving_avg_forecast','Moving Average Forecast')
calculateError(test_data.totalRequests,y_hat_mvng_avg.moving_avg_forecast,'Moving Average')

#4.Simple Exponential Smoothing (Simple Average+Wighted Moving Average) An equivalent ARIMA(0,1,1) model
#The forecast at time t+1 is equal to a weighted average between the most recent observation yt and the most recent forecast ŷ t|t−1

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
y_hat_exp_avg = test_data.copy()

#Build model
#alpha=0.6, Smoothing parameter
ses_model = SimpleExpSmoothing(np.asarray(train_data['totalRequests'])).fit(smoothing_level=0.1,optimized=False)
y_hat_exp_avg['SES'] = ses_model.forecast(len(test_data))
plotGraph(y_hat_exp_avg,'SES','Simple Exponential Smoothing')
calculateError(test_data.totalRequests,y_hat_exp_avg.SES,'Simple Exponential Smoothing')


#5.Holt’s Linear Trend method or Double exponential smoothing 
#(A method that takes into account the trend of the dataset) 
#An equivalent ARIMA(0,2,2) model
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

#Check trends & seasonality
seasonal_decompose(x=train_data['totalRequests'],freq=10).plot()

#Check for Stationarity
test_stationarity(test_data.totalRequests)
result = adfuller(train_data.totalRequests)

ts_log = np.log(test_data.totalRequests)
plt.plot(ts_log)
plt.plot(test_data.totalRequests)
train_data.plot()


#Build model
y_hat_holt_linear = test_data.copy()
holt_linear_model = Holt(np.asarray(train_data['totalRequests'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
y_hat_holt_linear['Holt_linear'] = ses_model.forecast(len(test_data))
plotGraph(y_hat_holt_linear,'Holt_linear','Holt Linear Trend')
calculateError(test_data.totalRequests,y_hat_holt_linear.Holt_linear,'Holt Linear')

#6. Holt-Winters Method or Triple exponential smoothing
#a method that takes into account both trend and seasonality to forecast
y_hat_holt_winter = test_data.copy()
holt_winter_model = ExponentialSmoothing(np.asarray(train_data['totalRequests']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
y_hat_holt_winter['Holt_Winter'] = holt_winter_model.forecast(len(test_data))
plotGraph(y_hat_holt_winter,'Holt_Winter','Holt Winter Trend')
calculateError(test_data.totalRequests,y_hat_holt_winter.Holt_Winter,'Holt Winter')

#7. SARIMA(p,d,q)(P,D,Q)s (Seasonal Autoregressive Integrated Moving average)
#ARIMA models aim to describe the correlations in the data with each other
from statsmodels.tsa.statespace.sarimax import SARIMAX
y_hat_sarima = test_data.copy()
sarima_model = SARIMAX(train_data.totalRequests, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit()
y_hat_sarima_result=pd.DataFrame()
y_hat_sarima_result['SARIMA'] = sarima_model.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, dynamic=True)
plotGraph(y_hat_sarima_result,'SARIMA','Seasonal Autoregressive Integrated Moving average')
calculateError(test_data.totalRequests,y_hat_sarima_result.SARIMA,'SARIMA')

#Plot Model Evaluation Metrics
evaluation_metrics.plot(y=['MAPE'],kind="bar",color='orange')
evaluation_metrics.plot(y=['RMSE'],kind="bar")

