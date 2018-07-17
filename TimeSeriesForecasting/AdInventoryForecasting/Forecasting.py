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

import warnings
warnings.filterwarnings("ignore")

train_data=pd.read_excel("/home/avinash/Learning/MachineLearning/TimeSeriesForecasting/AdInventoryForecasting/Dataset/forecasting_train_dataset_2016-18.xlsx")
test_data = pd.read_excel('/home/avinash/Learning/MachineLearning/TimeSeriesForecasting/AdInventoryForecasting/Dataset/forecasting_test_dataset.xlsx')

#Visualze Dataset
def visualize_dataset():
    #Total Requests
    plt.figure(figsize=(16,8))
    plt.plot(train_data.date, train_data['totalRequests'], label='Train_totalRequests')
    #plt.plot(test_data.date,test_data['totalRequests'], label='Test_totalRequests')
    plt.legend(loc='best')
    plt.title('Total Ad Requests Data')
    
    #Paid Impressions
    plt.figure(figsize=(16,8))
    plt.plot(train_data.date, train_data['paidImpressions'], label='Train_paidImpressions')
    #plt.plot(test_data.date,test_data['paidImpressions'], label='Test_paidImpressions')
    plt.legend(loc='best')
    plt.title('Total Paid Impressions Data')

visualize_dataset()

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
    
def check_seasonality_trend(train_series):
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    
    #Check trends & seasonality
    seasonal_decompose(x=train_series,freq=10).plot()
    
    #Check for Stationarity
    #test_stationarity(train_series)
    print(adfuller(train_data.totalRequests))
    
    
check_seasonality_trend(train_data.totalRequests)

#Create Evaluatiom Matrix
evaluation_metrics=pd.DataFrame()

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
def calculateError(input_data,predicted_data,model_name):
    rmse = sqrt(mean_squared_error(input_data, predicted_data))
    map_error=mape(input_data, predicted_data)
    accuracy=calculate_accuracy(input_data, predicted_data,10)
    print('RMSE='+str(rmse))
    print('MAPE='+str(map_error))
    print('Accuracy='+str(accuracy))
    evaluation_metrics.set_value(model_name,'RMSE',rmse)
    evaluation_metrics.set_value(model_name,'MAPE',map_error)
    evaluation_metrics.set_value(model_name,'Accuracy',accuracy)


def plotGraph(test_series,predicted_dataframe,predicted_column_name,method_label):
    plt.figure(figsize=(16,8))
    #plt.plot(train_data.date, train_data['totalRequests'], label='Train')
    plt.plot(test_data.date,test_series, label='Test')
    plt.plot(test_data.date,predicted_dataframe[predicted_column_name], label=method_label)
    plt.legend(loc='best')
    plt.title(method_label)
    plt.show()

#1. Naive Method
def naive_method(train_series,test_series):
    y_hat = pd.DataFrame(test_series)
    y_hat['Predicted'] = train_series.iloc[-1]
    plotGraph(test_series,y_hat,'Predicted','Naive Forecast')
    calculateError(test_series,y_hat.Predicted,'Naive')
    return y_hat

y_hat=naive_method(train_data.totalRequests,test_data.totalRequests)

#2. Simple Average
def simple_average_method(train_series,test_series):
    y_hat_avg = pd.DataFrame(test_series)
    avg_series= []
    training_series=list(train_series)
    for i in range(len(test_series)):
        mean_value=pd.Series(training_series).mean()
        #print('mean='+str(mean_value))
        avg_series.append(mean_value)
        training_series.append(mean_value)
    
    y_hat_avg['Predicted'] =avg_series  
    plotGraph(test_series,y_hat_avg,'Predicted','Average Forecast')
    calculateError(test_series,y_hat_avg.Predicted,'Simple Average')
    return y_hat_avg

y_hat_avg=simple_average_method(train_data.totalRequests,test_data.totalRequests)

#3. Moving Average Window/Sliding technique
def moving_average_method(train_series,test_series):
    y_hat_mvng_avg = pd.DataFrame(test_series)
    avg_series= []
    training_series=list(train_series)
    for i in range(len(test_series)):
        mean_value=pd.Series(training_series).rolling(7).mean().iloc[-1]
        #print('mean='+str(mean_value))
        avg_series.append(mean_value)
        training_series.append(mean_value)
    
    y_hat_mvng_avg['Predicted'] = avg_series
    plotGraph(test_series,y_hat_mvng_avg,'Predicted','Moving Average Forecast')
    calculateError(test_series,y_hat_mvng_avg.Predicted,'Moving Average')
    return y_hat_mvng_avg

y_hat_mvng_avg=moving_average_method(train_data.totalRequests,test_data.totalRequests)

#3.1 Weighted Moving Average technique
def weighted_moving_average_method(train_series,test_series):
    y_hat_wmvng_avg = pd.DataFrame(test_series)
    avg_series= []
    weights=np.random.dirichlet(np.ones(10),size=1)
    weights=pd.Series(weights[0]).sort_values(ascending=False)
    training_series=list(train_series)
    for i in range(len(test_series)):
        mean_value=np.average(np.asarray(training_series[len(training_series)-10:]),weights=weights)
        #print('mean='+str(mean_value))
        avg_series.append(mean_value)
        training_series.append(mean_value)
    y_hat_wmvng_avg['Predicted'] = avg_series
    plotGraph(test_series,y_hat_wmvng_avg,'Predicted','Weighted Moving Average Forecast')
    calculateError(test_series,y_hat_wmvng_avg.Predicted,'Weighted Moving Average')
    return y_hat_wmvng_avg
 
y_hat_wmvng_avg=weighted_moving_average_method(train_data.totalRequests,test_data.totalRequests)

#4.Simple Exponential Smoothing (Simple Average+Wighted Moving Average) An equivalent ARIMA(0,1,1) model
#The forecast at time t+1 is equal to a weighted average between the most recent observation yt and the most recent forecast ŷ t|t−1
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

def simple_exponential_smoothing_method(train_series,test_series):
    y_hat_exp_avg = pd.DataFrame(test_series)
    #Build model
    #alpha=0.6, Smoothing parameter
    ses_model = SimpleExpSmoothing(np.asarray(train_series)).fit()
    y_hat_exp_avg['Predicted'] = ses_model.forecast(len(test_series))
    plotGraph(test_series,y_hat_exp_avg,'Predicted','Simple Exponential Smoothing')
    calculateError(test_series,y_hat_exp_avg.Predicted,'Simple Exponential Smoothing')
    return y_hat_exp_avg

y_hat_exp_avg=simple_exponential_smoothing_method(train_data.totalRequests,test_data.totalRequests)

#5.Holt’s Linear Trend method or Double exponential smoothing 
#(A method that takes into account the trend of the dataset) 
#An equivalent ARIMA(0,2,2) model
def holt_linear_trend_method(train_series,test_series):
    y_hat_holt_linear = pd.DataFrame(test_series)
    holt_linear_model_fit = Holt(np.asarray(train_series)).fit()
    y_hat_holt_linear['Predicted'] = holt_linear_model_fit.forecast(len(test_series))
    plotGraph(test_series,y_hat_holt_linear,'Predicted','Holt Linear Trend')
    calculateError(test_series,y_hat_holt_linear.Predicted,'Holt Linear')
    return y_hat_holt_linear

y_hat_holt_linear=holt_linear_trend_method(train_data.totalRequests,test_data.totalRequests)

#6. Holt-Winters Method or Triple exponential smoothing
#A method that takes into account both trend and seasonality to forecast
def holt_winters_method(train_series,test_series):
    y_hat_holt_winter = pd.DataFrame(test_series)
    holt_winter_model = ExponentialSmoothing(np.asarray(train_series) ,seasonal_periods=15 ,trend='add', seasonal='add')
    holt_winter_model_fit=holt_winter_model.fit()
    y_hat_holt_winter['Predicted'] = holt_winter_model_fit.forecast(len(test_series))
    plotGraph(test_series,y_hat_holt_winter,'Predicted','Holt Winter Trend')
    calculateError(test_series,y_hat_holt_winter.Predicted,'Holt Winter')
    return y_hat_holt_winter

y_hat_holt_winter=holt_winters_method(train_data.totalRequests,test_data.totalRequests)

#Plot Model Evaluation Metrics
evaluation_metrics.plot(y=['MAPE'],kind="bar",color='orange')
evaluation_metrics.plot(y=['RMSE'],kind="bar")
evaluation_metrics.plot(y=['Accuracy'],kind="bar")

