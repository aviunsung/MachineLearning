#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:09:14 2018

@author: Avinash Pandey
"""

import pandas as pd
from sklearn import svm
from sklearn import metrics

train_dataset=pd.read_excel("/home/avinash/MachineLearning/AdQuality/Dataset/adquality_train_dataset.xlsx")
test_dataset = pd.read_excel('/home/avinash/MachineLearning/AdQuality/Dataset/AdQuality_additional_test_data.xlsx')

#exclude ucrid
train_dataset.drop(['ucrid'], axis=1,inplace=True)
test_dataset.drop(['ucrid'], axis=1,inplace=True)

X_train=train_dataset.iloc[:,0:7]
X_test=test_dataset.iloc[:,0:7]
y_train=train_dataset['manual_review']
y_test=test_dataset['manual_review']

def convertToLabels(dataframe):
    dataframe=dataframe.replace(to_replace=[-102, -101,-1,-2,-3,-5,-6, 1, 2, 6, 7],value=0)
    dataframe[dataframe != 0] = 1
    return dataframe

###....Data Preprocessing....
def data_preprocessing(X_train,y_train,X_test,y_test):
    #Handle Missing Data
    X_train['adsnoop'].fillna(0, inplace=True)
    X_train['confiant'].fillna(0, inplace=True)
    X_test['adsnoop'].fillna(0, inplace=True)
    X_test['confiant'].fillna(0, inplace=True)
    
    #Convert adsnoop,confiant and manual_review to 0 and 1
    X_test['adsnoop']=convertToLabels(X_test['adsnoop'])
    X_test['confiant']=convertToLabels(X_test['confiant'])
    y_test=convertToLabels(y_test)
    
    #Drop adsnoop and confiant
#    X_train.drop(['adsnoop'], axis=1,inplace=True)
#    X_train.drop(['confiant'], axis=1,inplace=True)
#    X_test.drop(['adsnoop'], axis=1,inplace=True)
#    X_test.drop(['confiant'], axis=1,inplace=True)

    
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
y_predicted=pd.DataFrame()
k=0

def evaluate_model(model_name,y_pred):
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
    model_accuracy.set_value(model_name,'roc_auc_score',roc_auc_score)
    print('roc_auc_score::',roc_auc_score)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred,pos_label=1)
    auc=metrics.auc(fpr, tpr)
    model_accuracy.set_value(model_name,'auc',auc)
    print('auc::',auc)
    average_precision_score=metrics.average_precision_score(y_test, y_pred)
    model_accuracy.set_value(model_name,'average_precision_score',average_precision_score)
    print('average_precision_score::',average_precision_score)
    
    print('Confusion Matrix::',metrics.confusion_matrix(y_test, y_pred,labels=[0, 1])) 
    #print('feature_importance::',model.feature_importances_)
    print(metrics.classification_report(y_test, y_pred)) 
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
    return evaluate_model(model_name,y_pred)
    

#-----Support Vector Classifier---#
# Create SVM classification object 
model = svm.SVC(kernel='linear', C=1,cache_size=400)
model_accuracy=build_and_test_model(model,'SVM_Linear',X_train,y_train,X_test,y_test)

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

#-----Random Forest (Bagging)--------#
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(n_estimators=200,criterion='gini',max_depth=7,max_features=2,n_jobs=-1)
model_accuracy=build_and_test_model(model,'Random Forest',X_train,y_train,X_test,y_test)


#-----Gradient Boosting (GBM)--------#
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100)
model_accuracy=build_and_test_model(model,'Gradient Boosting',X_train,y_train,X_test,y_test)


#-----eXtreme Gradient Boosting (XGBoost)-----#
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=100,n_jobs=-1)
model_accuracy=build_and_test_model(model,'XGBoost',X_train,y_train,X_test,y_test)

#-----Naive Bayes Classifier -------#
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model_accuracy=build_and_test_model(model,'Naive Bayes',X_train,y_train,X_test,y_test)

#--- Logistic Regression -------#
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')
model_accuracy=build_and_test_model(model,'Logistic Regression',X_train,y_train,X_test,y_test)

#----Recursive Feature Elimination ___#
from sklearn.feature_selection import RFE
rfe = RFE(model)
rfe = rfe.fit(X_train,y_train)
print(rfe.support_)
print(rfe.ranking_)

X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)
#cols=['dsp_id','geo_id','confiant']
cols=[0,3,6]
#cols=[0,3,4]

X_train_sliced=X_train.loc[:,cols]
X_test_sliced=X_test.loc[:,cols]

model_accuracy=build_and_test_model(model,'RFE Logistic Regression',X_train_sliced,y_train,X_test_sliced,y_test)

#----K-Nearest Neighbour (KNN) ___#
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3,n_jobs=-1)
model_accuracy=build_and_test_model(model,'KNN',X_train,y_train,X_test,y_test)

#------ Majority Vote of All Classifiers-----##
y_predicted_majority=y_predicted.mode(axis=1)
if(len (y_predicted_majority.columns)>1):
    y_predicted_majority.drop([1], axis=1,inplace=True)
model_accuracy= evaluate_model("Majority Vote",y_predicted_majority)
y_predicted.insert(loc=k,column='y_Majority Vote',value=y_predicted_majority)
k=k+1
y_predicted.insert(loc=k,column='y_test',value=y_test)

model_accuracy.plot.bar()
