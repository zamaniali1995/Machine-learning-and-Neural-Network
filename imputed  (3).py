#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 18:01:45 2019

@author: arian
"""
#importing packages
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier as Knearest
import sklearn.svm as svm
from sklearn.metrics import accuracy_score as acc
from sklearn.impute import SimpleImputer as imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix 
import matplotlib.pyplot as plt

def dataLoader(dir_dtrain_1, dir_dtrain_2, dir_data_test, dir_lable_test, merge_flag, is_train):
    if is_train:
        if merge_flag:
            data_set = pd.read_csv(dir_dtrain_1)
        else:
            data_set = pd.read_csv(dir_dtrain_1)
    else:
         data = pd.read_csv(dir_data_test)
         target = pd.read_csv(dir_lable_test)
         data_set = pd.concat([target, data], axis=1)
    data_set.columns=['ID', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 
                      'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 
                      'F17', 'F18', 'F19']
    lable=[]
    scaler = StandardScaler(copy=False)
    data_set = data_set.replace(0, np.NaN)
    imp = imputer(missing_values=np.NaN, strategy='mean')
# Drops rows with all nans
    drop_idx=[]
    nan_detected_data_set = pd.isnull(data_set)
    for i in range(data_set.shape[0]):
        cnt = 0
        for j in range(1, data_set.shape[1]):
            if nan_detected_data_set.iloc[i, j]:
                cnt += 1
        if cnt==data_set.shape[1]-1:
            drop_idx.append(i)
    data = data_set.drop(data_set.index[drop_idx])
    if is_train:
# Replaces nan features with mean and scales features
        for i in range(1,37):
            lable.append(data[data['ID']==i])
            lable[i-1]= imp.fit_transform(lable[i-1])
#            scaler.fit_transform(lable[i-1][:,1:])
            
        data= np.concatenate(lable)
        target, data = data[:, 0], data[:, 1:]
        np.savetxt('imputedTrain.csv', data, delimiter=',')
    else:
        data = imp.fit_transform(data)
#        scaler.fit_transform(data[:, 1:])
        target, data = data[:, 0], data[:, 1:]
        np.savetxt('imputedTest.csv', data, delimiter=',')
    return data, drop_idx, target


train_dir_1= './DataSet/Train_DB_1.csv'
train_dir_2 = './DataSet/Train_DB_2.csv'
test_data_dir = './DataSet/Test_Data.csv'
test_target_dir = './DataSet/Test_Labels.csv'
data_train, drop_idx_train, target_train = dataLoader(train_dir_1, train_dir_2, test_data_dir,
                                        test_target_dir, 1, 1)
data_test, drop_idx_test, target_test = dataLoader(train_dir_1, train_dir_2, test_data_dir,
                                      test_target_dir, 1, 0) 
#%%

#data_set_test = testLoader(valid_data_dir, valid_target_dir)
#%%
#data_test = np.loadtxt(open(valid_data_dir, "rb"), delimiter = ",", skiprows = 1)
#mask= np.any(np.equal(data_test,0), axis=1)
#data_test= data_test[~mask]
#target_test=np.loadtxt(open(valid_target_dir, "rb"), delimiter = ",", skiprows = 1)
#target_test= target_test[~mask]
#scaler = StandardScaler()
#data_test = scaler.fit_transform(data_test)
#%%
# Classification QDA
clf = qda()
clf.fit(data_train, target_train)
testPredict= clf.predict(data_test)
ans= acc(target_test, testPredict)
print("QDA ACC= ", ans)

# Classification LDA
clf2= lda()
clf2.fit(data_train,target_train)
predictLDA= clf2.predict(data_test)
ans2= acc(target_test, predictLDA)
print("LDA ACC= ", ans2)

# Classification Tree
clf3= tree.DecisionTreeClassifier()
clf3.fit(data_train,target_train)
predictTree= clf3.predict(data_test)
ans3= acc(target_test, predictTree)
print("Tree ACC= ", ans3)

# Classification KNN 
knn = Knearest(n_neighbors=2)
knn.fit(data_train,target_train)
predictKNN = knn.predict(data_test)
ans5= acc(target_test, predictKNN)
print("KNN ACC= ", ans5)


# Classification Kernel SVM
clf4 = svm.NuSVC(kernel = 'poly', gamma='scale')
clf4.fit(data_train, target_train)
predicteSVM = clf4.predict(data_test)
ans4 = acc(target_test, predicteSVM)
print ("kernel SVM ACC= ", ans4)

#%%

#merged = pd.read_csv('/home/arian/Desktop/stat learning/Home Work/assignment 3/DataSet/merged_NaN.csv')
#lable=[]
##scaler = StandardScaler()
#imp = imputer(missing_values=np.NaN, strategy='mean')
#for i in range(36):
#    lable.append(merged[merged['ID']==1])
#    lable[i]= imp.fit_transform(lable[i])
#    lable[i]= scaler.fit_transform(lable[i])
    
#with open('outputfile.csv', 'ab') as f:
#    for i in range(36):
#        np.savetxt(f, lable[i], delimiter=',')
    
