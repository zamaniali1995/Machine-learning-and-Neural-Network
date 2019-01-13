#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 18:01:45 2019

@author: arian
"""
#importing packages
import tensorflow as tf
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier as Knearest
import sklearn.svm as svm
from sklearn.metrics import accuracy_score as acc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix 
import matplotlib.pyplot as plt


def dataLoader(directory):
    scaler = StandardScaler()
    Data = np.loadtxt(open(directory, "rb"), delimiter=",", skiprows=1)
    mask= np.any(np.equal(Data,0), axis=1)
    Data= Data[~mask]
    target , data = Data[:,0] , Data[:,1:]
    data = scaler.fit_transform(data)
    return data, target


train_dir= './DataSet/Train_DB_1.csv'
data, target = dataLoader(train_dir)
#%%
valid_data_dir = './DataSet/Test_Data.csv'
valid_target_dir = './DataSet/Test_Labels.csv'
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
print(ans)

# Classification LDA
clf2= lda()
clf2.fit(data,target)
predictLDA= clf2.predict(validData)
ans2= acc(validTarget, predictLDA)
print(ans2)

# Classification Tree
clf3= tree.DecisionTreeClassifier()
clf3.fit(data,target)
predictTree= clf3.predict(validData)
ans3= acc(validTarget, predictTree)
print(ans3)

# Classification KNN (with error!!)
knn = Knearest(n_neighbors=2)
knn.fit(data,target)
predictKNN = knn.predict(validData)
#ans3= acc(validTarget, ans3)
#print(ans3)


# Classification Kernel SVM
clf4 = svm.NuSVC(kernel = 'sigmoid')
clf4.fit(data, target)
predicteSVM = clf4.predict(validData)
ans4 = acc(validTarget, predicteSVM)
print (ans4)
