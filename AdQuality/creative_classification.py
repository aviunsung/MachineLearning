#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:09:14 2018

@author: Avinash Pandey
"""

import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction import FeatureHasher

train_dataset=pd.read_excel("/home/avinash/personalRepo/MachineLearning/AdQuality/Dataset/adquality_train_dataset.xlsx")
test_dataset = pd.read_excel('/home/avinash/personalRepo/MachineLearning/AdQuality/Dataset/AdQuality_additional_test_data.xlsx')

#exclude ucrid
train_dataset.drop(['ucrid'], axis=1,inplace=True)
test_dataset.drop(['ucrid'], axis=1,inplace=True)

X_train=train_dataset.iloc[:,0:7]
X_test=test_dataset.iloc[:,0:7]
y_train=train_dataset['manual_review']
y_test=test_dataset['manual_review']

X_train.describe()
X_train.info()

def convertToLabels(dataframe):
    dataframe=dataframe.replace(to_replace=[0],value=1)
    dataframe=dataframe.replace(to_replace=[-102, -101,-3,-5,-6, 1, 2, 6, 7],value=0)
    dataframe[dataframe != 0] = 1
    return dataframe

def feature_hashing(dataset):
    fh = FeatureHasher(n_features=16, input_type='string')
    hashed_features = fh.fit_transform(dataset['landing_page_url'])
    hashed_features = hashed_features.toarray()
    hashed_features=pd.DataFrame(hashed_features)
    dataset=pd.concat([dataset,hashed_features],axis=1)
    dataset.drop(['landing_page_url'], axis=1,inplace=True)
    return dataset

###....Data Preprocessing....
def data_preprocessing(X_train,y_train,X_test,y_test):
    #Handle Missing Data
    X_train['adsnoop'].fillna(0, inplace=True)
    X_train['confiant'].fillna(0, inplace=True)
    X_train['landing_page_url'].fillna('NA', inplace=True)
    X_test['adsnoop'].fillna(0, inplace=True)
    X_test['confiant'].fillna(0, inplace=True)
    X_test['landing_page_url'].fillna('NA', inplace=True)
    
    #Convert adsnoop,confiant and manual_review to 0 and 1
    X_test['adsnoop']=convertToLabels(X_test['adsnoop'])
    X_test['confiant']=convertToLabels(X_test['confiant'])
    y_test=convertToLabels(y_test)
    
    #Drop adsnoop and confiant
#    X_train.drop(['adsnoop'], axis=1,inplace=True)
#    X_train.drop(['confiant'], axis=1,inplace=True)
#    X_test.drop(['adsnoop'], axis=1,inplace=True)
#    X_test.drop(['confiant'], axis=1,inplace=True)

    #------feature engineering--------#
    #Label Encoding
    from sklearn.preprocessing import LabelEncoder
    label_encoder=LabelEncoder()
    X_train['landing_page_url']=label_encoder.fit_transform(X_train['landing_page_url'].astype(str))
    X_test['landing_page_url']=label_encoder.fit_transform(X_test['landing_page_url'].astype(str))


    #Binary Encoding
#    encoder = ce.BinaryEncoder(cols=['landing_page_url'])
#    X_train = encoder.fit_transform(X_train)
#    X_test = encoder.fit_transform(X_test)

    # feature hashing #
#    X_train=feature_hashing(X_train)
#    X_test=feature_hashing(X_test)

    #One-Hot Encoding
#    from sklearn.preprocessing import LabelBinarizer
#    lb = LabelBinarizer()
#    lb_results = lb.fit_transform(X_train['landing_page_url'])
#    lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)
#    print(lb_results_df.head())
    
    #------feature scaling--------#
    from sklearn.preprocessing import MinMaxScaler
    sc_X=MinMaxScaler()
    X_train=sc_X.fit_transform(X_train)
    X_test=sc_X.fit_transform(X_test)
    
    #feature standardization (μ=0 and σ=1)
#    from sklearn.preprocessing import scale
#    X_train=scale(X_train)
#    X_test=scale(X_test)
#    
    return X_train,y_train,X_test,y_test

X_train,y_train,X_test,y_test=data_preprocessing(X_train,y_train,X_test,y_test)   

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
    

#-----Support Vector Classifier---#
# Create SVM classification object 
model = svm.SVC(kernel='linear', C=0.1,cache_size=400)
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
model = LogisticRegression(solver='liblinear',C=0.01)
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
cols=[0,5,6]
#cols=[0,3,4]

X_train_sliced=X_train.loc[:,cols]
X_test_sliced=X_test.loc[:,cols]

model_accuracy=build_and_test_model(model,'RFE Logistic Regression',X_train_sliced,y_train,X_test_sliced,y_test)

#----K-Nearest Neighbour (KNN) ___#
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=10,n_jobs=-1)
model_accuracy=build_and_test_model(model,'KNN',X_train,y_train,X_test,y_test)

#------ Majority Vote of All Classifiers-----##
y_predicted_majority=y_predicted.mode(axis=1)
if(len (y_predicted_majority.columns)>1):
    y_predicted_majority.drop([1], axis=1,inplace=True)
model_accuracy= evaluate_model("Majority Vote",y_test,y_predicted_majority)
y_predicted.insert(loc=k,column='y_Majority Vote',value=y_predicted_majority)
k=k+1
y_predicted.insert(loc=k,column='y_test',value=y_test)

## Plot Accuracy Report ##
model_accuracy.plot.bar()

## Plot roc_auc_score Graph ##
model_accuracy['roc_auc_score'].plot.bar()

