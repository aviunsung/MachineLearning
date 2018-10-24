#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 22:09:14 2018

@author: Avinash Pandey
"""

import pandas as pd
from sklearn import svm
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
model_accuracy=pd.DataFrame()

def build_and_test_model(model,model_name,X_train,y_train,X_test,y_test):
    global model_accuracy
    model.fit(X_train,y_train)
    #model.score(X_train,y_train)
    y_pred=model.predict(X_test)
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
    print('Confusion Matrix::',metrics.confusion_matrix(y_test, y_pred,labels=[0, 1])) 
    print('feature_importance::',model.feature_importances_)
    print(metrics.classification_report(y_test, y_pred)) 
    return model_accuracy

#-----Support Vector Classifier---#
# Create SVM classification object 
model = svm.SVC(kernel='linear', C=1,cache_size=400)
#model_accuracy=build_and_test_model(model,'SVM_Linear',X_train,y_train,X_test,y_test)

#model = svm.SVC(kernel='poly', degree=8,cache_size=400)
#model_accuracy=build_and_test_model(model,'SVM_Poly',X_train,y_train,X_test,y_test)
#
#model = svm.SVC(kernel='rbf')
#model_accuracy=build_and_test_model(model,'SVM_Rbf',X_train,y_train,X_test,y_test)
#
#model = svm.SVC(kernel='sigmoid')
#model_accuracy=build_and_test_model(model,'SVM_Sigmoid',X_train,y_train,X_test,y_test)

#----Decision Tree-----#
from sklearn import tree
model = tree.DecisionTreeClassifier(criterion='entropy',max_depth=7,max_features=2)
model_accuracy=build_and_test_model(model,'Decision Tree',X_train,y_train,X_test,y_test)

#-----Random Forest--------#
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(n_estimators=2000,criterion='gini',max_depth=7,max_features=2,n_jobs=6)
model_accuracy=build_and_test_model(model,'Random Forest',X_train,y_train,X_test,y_test)

#--HyperParameters Tuning--##
