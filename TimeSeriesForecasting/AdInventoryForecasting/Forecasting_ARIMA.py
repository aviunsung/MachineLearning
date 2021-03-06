#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 22:09:47 2018

@author: Avinash Pandey
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

train_data=pd.read_excel("/home/avinash/Learning/MachineLearning/TimeSeriesForecasting/AdInventoryForecasting/Dataset/forecasting_train_dataset.xlsx")
test_data = pd.read_excel('/home/avinash/Learning/MachineLearning/TimeSeriesForecasting/AdInventoryForecasting/Dataset/forecasting_test_dataset.xlsx')

train_series=train_data.totalRequests
test_series=test_data.totalRequests

#train_series=train_data.paidImpressions
#test_series=test_data.paidImpressions

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
    plt.figure(figsize=(12,4))
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
    test_stationarity(train_series)
    print(adfuller(train_data.totalRequests))
    
#Check trends & seasonality
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

def plot_acf_pacf(data,lag):
    from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
    plot_acf(data, lags=lag)
    plot_pacf(data, lags=lag)
    plt.show()
    
    from pandas.tools.plotting import autocorrelation_plot
    autocorrelation_plot(train_data.totalRequests)
    plt.show()
    
#AR -Auto Regressive Time series
#x(t) = alpha *  x(t – 1) + error (t)

#####Data Preprocessing##################
####Eliminating Trend & Seasonality###########

#A. Differencing
train_data_diff=pd.DataFrame()
train_data_diff['date']=train_data.date
train_data_diff['totalRequests']=train_data.totalRequests-train_data.totalRequests.shift(7)
train_data_diff['paidImpressions']=train_data.paidImpressions-train_data.paidImpressions.shift(1)
train_data_diff.dropna(inplace=True)
test_stationarity(train_data_diff.totalRequests)
plot_acf_pacf(train_data.totalRequests,30)
plot_acf_pacf(train_data_diff.totalRequests,30)

# =============================================================================
# #B.Decomposing
# from statsmodels.tsa.seasonal import seasonal_decompose
# decomposition = seasonal_decompose(np.asarray(train_data.totalRequests),freq=10)
# decomposition.plot()
# plt.show()
# train_data_decompose=pd.DataFrame()
# train_data_decompose['date']=train_data.date
# train_data_decompose['totalRequests'] = decomposition.resid
# train_data_decompose.dropna(inplace=True)
# check_seasonality_trend(train_data_decompose)
# 
# =============================================================================
 #Check for Autocorrelation 
 #(correlation is calculated between the variable and itself at previous time steps)
from pandas.tools.plotting import lag_plot
lag_plot(train_data.totalRequests)
plt.show()
 

 #ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(train_data.totalRequests, nlags=20)
lag_pacf = pacf(train_data.totalRequests, nlags=20, method='ols')
 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train_data.totalRequests)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_data.totalRequests)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
 
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train_data.totalRequests)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_data.totalRequests)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

#1.Autoregression Model (AR(p)) 
#x(t) = alpha *  x(t – 1) + error (t)
#An autoregression model is a linear regression model that uses lagged variables as input variables
#It says current series values depend on its previous values with some lag (or several lags),which is referred to as p
from statsmodels.tsa.arima_model import ARIMA
def auto_regression_AR(train_series,test_series):
    y_hat_AR = pd.DataFrame(test_series)
    model_AR = ARIMA(train_series, order=(1, 1, 0))  
    model_AR_fit = model_AR.fit(disp=0) 
    #print(model_AR_fit.summary())
    y_hat_AR['Predicted']=model_AR_fit.forecast(steps=len(test_series))[0]
    plotGraph(test_series,y_hat_AR,'Predicted','Autoregression(AR)')
    calculateError(test_series,y_hat_AR.Predicted,'Autoregression(AR)')
    return y_hat_AR
    
y_hat_AR=auto_regression_AR(train_series,test_series)

#2.Moving Average (MA(q))
#x(t) = beta *  error(t-1) + error (t)
#It says that current error depends on the previous with some lag, which is referred to as q

def moving_average_MA(train_series,test_series):
    y_hat_MA = pd.DataFrame(test_series)
    model_MA = ARIMA(train_series, order=(0, 1, 7))  
    model_MA_fit = model_MA.fit(disp=-1) 
    #print(model_AR_fit.summary())
    y_hat_MA['Predicted']=model_MA_fit.forecast(steps=len(test_series))[0]
    plotGraph(test_series,y_hat_MA,'Predicted','Moving Average(MA)')
    calculateError(test_series,y_hat_MA.Predicted,'Moving Average(MA)')
    return y_hat_MA
    
y_hat_MA=moving_average_MA(train_series,test_series)

#3.Autoregressive Integrated Moving average (ARIMA(p,d,q))
#d=the number of past time points to subtract from the current value(differencing)

def ARIMA_method(train_series,test_series):
    y_hat_ARIMA = pd.DataFrame(test_series)
    model_ARIMA = ARIMA(train_series, order=(3, 1, 4))  
    model_ARIMA_fit = model_ARIMA.fit(disp=-1)
    print(model_ARIMA_fit.summary())
    y_hat_ARIMA['Predicted']=model_ARIMA_fit.forecast(steps=len(test_series))[0]
    plotGraph(test_series,y_hat_ARIMA,'Predicted','ARIMA')
    calculateError(test_series,y_hat_ARIMA.Predicted,'ARIMA')
    return y_hat_ARIMA

y_hat_ARIMA=ARIMA_method(train_series,test_series)

#4. SARIMA(p,d,q)(P,D,Q)s (Seasonal Autoregressive Integrated Moving average)
#ARIMA models aim to describe the correlations in the data with each other

def SARIMA_method(train_series,test_series):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y_hat_SARIMA = pd.DataFrame()
    model_SARIMA = SARIMAX(train_series, order=(1, 1, 4),seasonal_order=(2,1,4,4)).fit(optimize=True)  
    print(model_SARIMA.summary())
    y_hat_SARIMA['Predicted']=model_SARIMA.predict(start=len(train_series), end=len(train_series)+len(test_series)-1, dynamic=True)
    plotGraph(test_series,y_hat_SARIMA,'Predicted','SARIMA')
    calculateError(test_series,y_hat_SARIMA.Predicted,'SARIMA')
    return y_hat_SARIMA

y_hat_SARIMA=SARIMA_method(train_series,test_series)


#Plot Model Evaluation Metrics
evaluation_metrics.plot(y=['MAPE'],kind="bar",color='orange')
evaluation_metrics.plot(y=['RMSE'],kind="bar")
evaluation_metrics.plot(y=['Accuracy'],kind="bar",color='green')


