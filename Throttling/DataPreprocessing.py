#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:28:30 2019

@author: Avinash Pandey
"""

import pandas as pd
import seaborn as sns

train_dataset=pd.read_csv("/home/avinash/personalRepo/MachineLearning/Throttling/train_dataset/Hour1_30thJan/156307.tsv",delimiter='\t',encoding='utf-8')
test_dataset = pd.read_csv('/home/avinash/personalRepo/MachineLearning/Throttling/test_dataset/156307.tsv',delimiter='\t',encoding='utf-8')

train_dataset_raw = train_dataset.copy()  # Save original data set, just in case.

# Descriptive statistics
train_dataset.describe()
len(train_dataset)
train_dataset['verified_digit'].value_counts()
train_dataset['gctry'].value_counts().head(1).index.values[0]

# Data types
train_dataset.dtypes

# Create table for missing data analysis
def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = ((df.isnull().sum()/len(df))*100).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data)
    return missing_data

#Data Preparation/Data Cleaning
def data_preprocessing(train_dataset):
    #1. Handle Missing Values
    train_dataset.isnull().sum()
    draw_missing_data_table(train_dataset)
    
    ##Drop null values
    #train_dataset=train_dataset.dropna(subset=['gctry']) 
    train_dataset.drop(['id','pi','dab','ab','ai','ut','wcid'],axis=1, inplace=True)
    train_dataset['md'].fillna('NA')
    train_dataset=train_dataset.dropna()
    draw_missing_data_table(train_dataset) 
    len(train_dataset)
    
    means=train_dataset.groupby('gctry')['verified_digit'].count()
    
    sns.barplot(x=train_dataset['verified_digit'].value_counts().index,y=train_dataset['verified_digit'].value_counts().values)
    
    ##Feature extraction
    # =============================================================================
    # for index in train_dataset.columns:
    #     print(train_dataset.groupby(str(index))['verified_digit'].count())
    # 
    # =============================================================================
    
    
    ##Feature selection
    train_dataset.drop(['pid','hour','sid','pfi','uh','je','los','oi','di','adtype'],axis=1, inplace=True)
    
    X_train=train_dataset.iloc[:,0:11]
    X_train['pb']=train_dataset['pb']
    
    ##Feature scaling/transformation
    import category_encoders as ce
    columns=['ts','adid','gctry','mk','md','os','lo']
    
    #Label Encoding
    ce_ord=ce.OrdinalEncoder(cols=columns)
    X_train = ce_ord.fit_transform(X_train)
    y_train=train_dataset['verified_digit']
    
    #Binary Encoding
# =============================================================================
#     b_encoder = ce.BinaryEncoder(cols=columns)
#     X_train = b_encoder.fit_transform(X_train)
#     
# =============================================================================
    return X_train,y_train

X_train,y_train=data_preprocessing(train_dataset)
X_test,y_test=data_preprocessing(test_dataset)

#------feature scaling--------#
from sklearn.preprocessing import MinMaxScaler
sc_X=MinMaxScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
    

from sklearn import metrics
model_accuracy=pd.DataFrame()
y_predicted=pd.DataFrame()
k=0

def evaluate_model(model_name,y_test,y_pred):
    print('Evaluating Model::',model_name)
    global model_accuracy
    # Model Accuracy:
    accuracy=metrics.accuracy_score(y_test, y_pred)*100
    model_accuracy.set_value(model_name,'Accuracy',accuracy)
    print('accuracy::',accuracy)
    # Model Precision:
    precision=metrics.precision_score(y_test, y_pred)*100
    model_accuracy.set_value(model_name,'Precision',precision)
    print('precision::',precision)
    # Model Recall:
    recall=metrics.recall_score(y_test, y_pred)*100
    model_accuracy.set_value(model_name,'Recall',recall)
    print('recall::',recall)
    f1_score=metrics.f1_score(y_test, y_pred)
    model_accuracy.set_value(model_name,'f1_score',f1_score)
    print('f1_score::',f1_score)
    roc_auc_score=metrics.roc_auc_score(y_test, y_pred)
    model_accuracy.set_value(model_name,'roc_auc_score',roc_auc_score*100)
    print('roc_auc_score::',roc_auc_score)
    #fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred,pos_label=1)
    #auc=metrics.auc(fpr, tpr)
    #model_accuracy.set_value(model_name,'auc',auc)
    #print('auc::',auc)
    average_precision_score=metrics.average_precision_score(y_test, y_pred)
    model_accuracy.set_value(model_name,'average_precision_score',average_precision_score)
    print('average_precision_score::',average_precision_score)
    
    print('Confusion Matrix::',metrics.confusion_matrix(y_test, y_pred,labels=[0, 1])) 
    #print('feature_importance::',model.feature_importances_)
    print(metrics.classification_report(y_test, y_pred,)) 
    return model_accuracy

def build_and_test_model(model,model_name,X_train,y_train,X_test,y_test):
    print('Building and Testing Model::',model_name)
    global y_predicted
    global k
    model.fit(X_train,y_train)
    #model.score(X_train,y_train)
    y_pred=model.predict(X_test)
    y_predicted.insert(loc=k,column='y_'+model_name,value=y_pred)
    k=k+1
    return evaluate_model(model_name,y_test,y_pred)
    

#----Decision Tree-----#
from sklearn import tree
model = tree.DecisionTreeClassifier(criterion='entropy',max_depth=7,max_features=2)
model_accuracy=build_and_test_model(model,'Decision Tree',X_train,y_train,X_test,y_test)
