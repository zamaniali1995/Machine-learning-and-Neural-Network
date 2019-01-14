#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 18:01:45 2019

@author: arian
"""
#importing packages
import math
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
    scaler = StandardScaler()
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
# Replaces nan features with mean and scales features
    for i in range(1,37):
        lable.append(data[data['ID']==i])
        lable[i-1]= imp.fit_transform(lable[i-1])
        lable[i-1]= scaler.fit_transform(lable[i-1])
#    if is_train:
#       with open('imputedTrain.csv', 'ab') as f:
#           for i in range(36):
#               np.savetxt(f, lable[i], delimiter=',')
#           data = pd.read_csv('imputedTrain.csv')
#    else:
#       with open('imputedTest.csv', 'ab') as f:
#           for i in range(36):  
#               np.savetxt(f, lable[i], delimiter=',')s
#           data = pd.read_csv('imputedTest.csv')
    return data, drop_idx, lable


train_dir_1= './DataSet/Train_DB_1.csv'
train_dir_2 = './DataSet/Train_DB_2.csv'
test_data_dir = './DataSet/Test_Data.csv'
test_target_dir = './DataSet/Test_Labels.csv'
data_train, drop_idx_train, lable_train = dataLoader(train_dir_1, train_dir_2, test_data_dir,
                                        test_target_dir, 1, 1)
data_test, drop_idx_test, lable_test = dataLoader(train_dir_1, train_dir_2, test_data_dir,
                                      test_target_dir, 1, 0) 
#%%

data_set_test = testLoader(valid_data_dir, valid_target_dir)
#%%
validData = np.loadtxt(open(valid_data_dir, "rb"), delimiter = ",", skiprows = 1)
mask= np.any(np.equal(validData,0), axis=1)
validData= validData[~mask]
validTarget=np.loadtxt(open(valid_target_dir, "rb"), delimiter = ",", skiprows = 1)
validTarget= validTarget[~mask]
scaler = StandardScaler()
validData = scaler.fit_transform(validData)
#%%
# Classification QDA
clf = qda()
clf.fit(data,target)
testPredict= clf.predict(validData)
ans= acc(validTarget, testPredict)
print("QDA ACC= ", ans)

# Classification LDA
clf2= lda()
clf2.fit(data,target)
predictLDA= clf2.predict(validData)
ans2= acc(validTarget, predictLDA)
print("LDA ACC= ", ans2)

# Classification Tree
clf3= tree.DecisionTreeClassifier()
clf3.fit(data,target)
predictTree= clf3.predict(validData)
ans3= acc(validTarget, predictTree)
print("Tree ACC= ", ans3)

# Classification KNN 
knn = Knearest(n_neighbors=2)
knn.fit(data,target)
predictKNN = knn.predict(validData)
ans5= acc(validTarget, predictKNN)
print("KNN ACC= ", ans5)


# Classification Kernel SVM
clf4 = svm.NuSVC(kernel = 'sigmoid')
clf4.fit(data, target)
predicteSVM = clf4.predict(validData)
ans4 = acc(validTarget, predicteSVM)
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
    
