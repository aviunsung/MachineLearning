#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 22:09:14 2018

@author: Avinash Pandey
"""

import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import metrics

train_dataset=pd.read_excel("/home/avinash/MachineLearning/AdQuality/Dataset/adquality_train_dataset.xlsx")
test_dataset = pd.read_excel('/home/avinash/MachineLearning/AdQuality/Dataset/adquality_test_dataset.xlsx')

#exclude ucrid by slicicng
#train_dataset=train_data.iloc[:,1:9]
#test_dataset=test_data.iloc[:,1:9]

train_dataset.drop(['ucrid'], axis=1,inplace=True)
test_dataset.drop(['ucrid'], axis=1,inplace=True)


X_train=train_dataset.iloc[:,0:7]
X_test=test_dataset.iloc[:,0:7]
y_train=train_dataset['manual_review']
y_test=test_dataset['manual_review']

plt.scatter(x=X_train,y=y_train)
###....Data Preprocessing....
def data_preprocessing(X_train,y_train,X_test,y_test):
    #Handle Missing Data
    X_train['adsnoop'].fillna(0, inplace=True)
    X_train['confiant'].fillna(0, inplace=True)
    X_test['adsnoop'].fillna(0, inplace=True)
    X_test['confiant'].fillna(0, inplace=True)
    
    #Handle Categorical Data
    from sklearn.preprocessing import LabelEncoder
    label_encoder=LabelEncoder()
    X_train['landing_page_url']=label_encoder.fit_transform(X_train['landing_page_url'].astype(str))
    X_test['landing_page_url']=label_encoder.fit_transform(X_test['landing_page_url'].astype(str))

    #feature scaling
    from sklearn.preprocessing import StandardScaler
    sc_X=StandardScaler()
    X_train=sc_X.fit_transform(X_train)
    X_test=sc_X.fit_transform(X_test)
    return X_train,y_train,X_test,y_test

X_train,y_train,X_test,y_test=data_preprocessing(X_train,y_train,X_test,y_test)   
model_accuracy=pd.DataFrame(columns=['Model','Precision','Recall','Accuracy','F1 Score'])

def build_and_test_model(model,model_name,X_train,y_train,X_test,y_test,model_accuracy):
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    # Model Accuracy:
    accuracy=metrics.accuracy_score(y_test, y_pred)*100
    print('accuracy::',accuracy)
    # Model Precision:
    precision=metrics.precision_score(y_test, y_pred)*100
    print('precision::',precision)
    # Model Recall:
    recall=metrics.recall_score(y_test, y_pred)*100
    print('recall::',recall)
    f1_score=metrics.f1_score(y_test, y_pred)
    print('f1_score::',f1_score)
     
    model_accuracy.append(pd.Series([model_name,precision,recall,accuracy,f1_score],index=['Model','Precision','Recall','Accuracy','F1 Score']),ignore_index=True)
    print('Confusion Matrix::',metrics.confusion_matrix(y_test, y_pred,labels=[0, 1]))  
   # print(metrics.classification_report(y_test, y_pred)) 
    return model_accuracy

# Create SVM classification object 
model = svm.SVC(kernel='linear', C=10,cache_size=4500)
model_accuracy=build_and_test_model(model,'SVM_Linear',X_train,y_train,X_test,y_test,model_accuracy)

#model = svm.SVC(kernel='poly', degree=8,cache_size=4500)
#model_accuracy=build_and_test_model(model,'SVM_Poly',X_train,y_train,X_test,y_test,model_accuracy)
#
#model = svm.SVC(kernel='rbf')
#model_accuracy=build_and_test_model(model,'SVM_Rbf',X_train,y_train,X_test,y_test,model_accuracy)
#
#model = svm.SVC(kernel='sigmoid')
#model_accuracy=build_and_test_model(model,'SVM_Sigmoid',X_train,y_train,X_test,y_test,model_accuracy)

