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
    #plt.plot(train_data.index, train_data['totalRequests'], label='Train')
    plt.plot(test_data.index,test_data['totalRequests'], label='Test')
    plt.plot(predicted_dataframe.index,predicted_dataframe[column_name], label=method_label)
    plt.legend(loc='best')
    plt.title(method_label)
    plt.show()

def test_stationarity(timeseries_data):
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries_data, window=14)
    rolstd = pd.rolling_std(timeseries_data, window=14)

    #Plot rolling statistics:
    plt.figure(figsize=(16,8))
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
plt.figure(figsize=(12,8))
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

#AR -Auto Regressive Time series
#x(t) = alpha *  x(t – 1) + error (t)

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

#Check trends & seasonality
seasonal_decompose(x=train_data['totalRequests'],freq=10).plot()

#Check for Stationarity
test_stationarity(test_data.totalRequests)
result = adfuller(train_data.totalRequests)
print(result)

#####Data Preprocessing##################
####Eliminating Trend & Seasonality###########

#A. Differencing
train_data_diff=pd.DataFrame()
train_data_diff['date']=train_data.date
train_data_diff['totalRequests']=train_data.totalRequests-train_data.totalRequests.shift()
train_data_diff['paidImpressions']=train_data.paidImpressions-train_data.paidImpressions.shift()
train_data_diff.dropna(inplace=True)
test_stationarity(train_data_diff.totalRequests)

#B.Decomposing
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(np.asarray(train_data.totalRequests),freq=10)
decomposition.plot()
plt.show()
train_data_decompose=pd.DataFrame()
train_data_decompose['date']=train_data.date
train_data_decompose['totalRequests'] = decomposition.resid
train_data_decompose.dropna(inplace=True)

#Check for Autocorrelation 
#(correlation is calculated between the variable and itself at previous time steps)
from pandas.tools.plotting import lag_plot
lag_plot(train_data.totalRequests)
plt.show()

from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(train_data.totalRequests)
plt.show()

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(train_data.totalRequests, lags=31)
plot_pacf(train_data.totalRequests, lags=31)
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
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA

#Use AR
y_hat_AR = pd.DataFrame()
model = AR(train_data.totalRequests)
model_fit = model.fit()
#p=no of lag variables used (order of AR)
#alpha=coefficients used with lag variables
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
y_hat_AR['predictions'] = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, dynamic=False)
plotGraph(y_hat_AR,'predictions','Autoregression')
calculateError(test_data.totalRequests,y_hat_AR.predictions,'Autoregression')

#Use ARIMA
model_AR = ARIMA(train_data.totalRequests, order=(15, 1, 0))  
model_AR_fit = model_AR.fit(disp=0) 
print(model_AR_fit.summary())
y_hat_AR['predictions']=model_AR_fit.forecast(steps=len(test_data))[0]
#y_hat_AR['predictions'] = model_AR.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, dynamic=False)
plotGraph(y_hat_AR,'predictions','Autoregression(AR)')
calculateError(test_data.totalRequests,y_hat_AR.predictions,'Autoregression(AR)')


#2.Moving Average (MA(q))
#x(t) = beta *  error(t-1) + error (t)
#It says that current error depends on the previous with some lag, which is referred to as q
y_hat_MA = pd.DataFrame()
model_MA = ARIMA(train_data.totalRequests, order=(0, 1, 7))  
model_MA_fit = model_MA.fit(disp=-1)  
print(model_MA_fit.summary())
y_hat_MA['predictions']=model_MA_fit.forecast(steps=len(test_data))[0]
plotGraph(y_hat_MA,'predictions','Moving Average(MA)')
calculateError(test_data.totalRequests,y_hat_MA.predictions,'Moving Average(MA)')

#Predict Paid Impressions
model_ARIMA = ARIMA(train_data.paidImpressions, order=(0, 1, 1))  
model_ARIMA_fit = model_ARIMA.fit(disp=-1)
test_data['Predicted_Paid_Impressions']=model_ARIMA_fit.forecast(steps=len(test_data))[0]
calculateError(test_data.paidImpressions,test_data.Predicted_Paid_Impressions,'ARIMA')

#3.Autoregressive Integrated Moving average (ARIMA(p,d,q))
#d=the number of past time points to subtract from the current value(differencing)
y_hat_ARIMA = pd.DataFrame()
model_ARIMA = ARIMA(train_data.totalRequests, order=(15, 1, 7))  
model_ARIMA_fit = model_ARIMA.fit(disp=-1)
print(model_ARIMA_fit.summary())
y_hat_ARIMA['predictions']=model_ARIMA_fit.forecast(steps=len(test_data))[0]
test_data['Predicted_Total_Requests']=y_hat_ARIMA['predictions']
plotGraph(y_hat_ARIMA,'predictions','ARIMA')
calculateError(test_data.totalRequests,y_hat_ARIMA.predictions,'ARIMA')

#Predict Paid Impressions
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
#Train model
lm.fit(pd.DataFrame(train_data.totalRequests),pd.DataFrame(train_data.paidImpressions))
#Test model
predictions=lm.predict(pd.DataFrame(test_data.totalRequests))
test_data['Predicted_Paid_Impressions']=predictions
calculateError(test_data.paidImpressions,predictions,'Linear Regression')

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

