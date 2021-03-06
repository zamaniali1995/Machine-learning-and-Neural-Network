#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 18:01:45 2019

@author: ali and aryan
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
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 
numClass = 36
# dataLoder function: inputs: dirTrain1, dirTrain2, dirTestData, dirTestTarget, mergeFlag, isTrain, scale
#                     outputs: data, dropIdx, target
#                     if mergeFlage = 1 then datas in two directories dirTrain1 and dirTrain2 merge together 
#                     isTrain = 1 is for loading train dataset and isTrain = 0 is for loading test dataset.
#                     if scale = 1 then data will scale otherwise not.
def dataLoader(dirTrain1, dirTrain2, dirTestData, dirTestTarget, mergeFlag, isTrain, scale):
    if isTrain:
        if mergeFlag:
            dataSet1 = pd.read_csv(dirTrain1)
            dataSet2 = pd.read_csv(dirTrain2)
            dataSet = pd.concat([dataSet1, dataSet2])
        else:
            dataSet = pd.read_csv(dirTrain1)
    else:
         data = pd.read_csv(dirTestData)
         target = pd.read_csv(dirTestTarget)
         dataSet = pd.concat([target, data], axis=1)
         
    dataSet.columns=['ID', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 
                      'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 
                      'F17', 'F18', 'F19']
    lable=[]
    scaler = StandardScaler(copy=False)
    dataSet = dataSet.replace(0, np.NaN)
    imp = imputer(missing_values=np.NaN, strategy='mean')
# Drops rows with all nans
    dropIdx=[]
    nanDetected = pd.isnull(dataSet)
    for i in range(dataSet.shape[0]):
        cnt = 0
        for j in range(1, dataSet.shape[1]):
            if nanDetected.iloc[i, j]:
                cnt += 1
        if cnt==dataSet.shape[1]-1:
            dropIdx.append(i)
    data = dataSet.drop(dataSet.index[dropIdx])
    if isTrain:
# Replaces nan features with mean and scales features
        for i in range(1, numClass+1):
            lable.append(data[data['ID']==i])
            lable[i-1]= imp.fit_transform(lable[i-1])
            if scale:
                scaler.fit_transform(lable[i-1][:,1:])
            
        data = np.concatenate(lable)
        data = shuffle(data)
        target, data = data[:, 0], data[:, 1:]
        np.savetxt('imputedTrain.csv', data, delimiter=',')
    else:
#        data = imp.fit_transform(data)
#        data = data.replace(np.NaN, 0)
        
        
#        if scale:
#            scaler.fit_transform(data[:, 1:])
        target, data = data[:, 0], data[:, 1:]
#        np.savetxt('imputedTest.csv', data, delimiter=',')
    return data, dropIdx, target, lable
# Dataset directories
dirTrain1 = '../DataSet/Train_DB_1.csv'
dirTrain2 = '../DataSet/Train_DB_2.csv'
dirTestData = '../DataSet/Test_Data.csv'
dirTestTarget = '../DataSet/Test_Labels.csv'
# Loading train and test data
dataTrain, dropIdxTrain, targetTrain , label= dataLoader(dirTrain1, dirTrain2, dirTestData,
                                        dirTestTarget, 0, 1, 0)
#dataTest, dropIdxTest, targetTest, _ = dataLoader(dirTrain1, dirTrain2, dirTestData,
#                                        dirTestTarget, 0, 0, 0)  
# Loading test data
dataTest = pd.read_csv(dirTestData) 
targetTest = pd.read_csv(dirTestTarget)
dataTest = dataTest.replace(0, np.NaN)
nonDetected = pd.isnull(dataTest)
#dataTest = pd.concat([target, data], axis=1)

#%%
dataTestModified = []
for i in range(numClass):
    tmp = dataTest
    avg = np.mean(label[i], axis=0)
    for j in range(dataTest.shape[0]):
        for k in range(dataTest.shape[1]):
            if nonDetected.iloc[j, k]:
                tmp.iloc[j, k] = avg[k + 1]
    dataTestModified.append(tmp)
#%%
numOfTrCla = []
for i in range(numClass):
    numOfTrCla.append(label[i].shape[0])
plt.plot(numOfTrCla, 'r')
plt.ylabel("Number of Classes")
plt.xlabel("Classes")
plt.savefig('../Plots/numberOfTrainClasses.pdf')
#%%
qdaList = []
ldaList = []
treeList = []
KNNList = []
SVMList = []
for i in range(numClass):
    dataTest = dataTestModified[i]

    # Classification QDA
    clf = qda()
    clf.fit(dataTrain, targetTrain)
    trainPredic = clf.predict(dataTrain)
#    print("QDA Train ACC= ", acc(targetTrain, trainPredic))
    predicteQDA= clf.predict(dataTest)
    ans1 = acc(targetTest, predicteQDA)
    qdaList.append(ans1)
#    print("QDA Test ACC= ", ans1)
    
    # Classification LDA
    clf2= lda()
    clf2.fit(dataTrain,targetTrain)
    trainPredic = clf.predict(dataTrain)
    print("\nLDA train ACC= ", acc(targetTrain, trainPredic))
    predictLDA= clf2.predict(dataTest)
    ans2 = acc(targetTest, predictLDA)
    ldaList.append(ans2) 
#    print("LDA test ACC= ", ans2)
    
    # Classification Tree
    clf3= tree.DecisionTreeClassifier()
    clf3.fit(dataTrain,targetTrain)
    trainPredic = clf.predict(dataTrain)
#    print("\nTree train ACC= ", acc(targetTrain, trainPredic))
    predictTree= clf3.predict(dataTest)
    ans3= acc(targetTest, predictTree)
    treeList.append(ans3)
#    print("Tree test ACC= ", ans3)
    
    # Classification KNN 
    knn = Knearest(n_neighbors=2)
    knn.fit(dataTrain,targetTrain)
    trainPredic = clf.predict(dataTrain)
#    print("\nKnn train ACC= ", acc(targetTrain, trainPredic))
    predictKNN = knn.predict(dataTest)
    ans4= acc(targetTest, predictKNN)
    KNNList.append(ans4)
#    print("KNN test ACC= ", ans4)
    
    
    # Classification Kernel SVM
    #clf4 = svm.NuSVC(kernel = 'sigmoid', coef0=0.1, gamma='scale')
    #clf4 = svm.NuSVC(kernel = 'rbf', gamma=0.5)
    clf4 = svm.NuSVC(kernel = 'poly',coef0=120 ,gamma='scale')
    #clf4 = svm.NuSVC(kernel = 'linear', gamma='scale')
    clf4.fit(dataTrain, targetTrain)
    trainPredic = clf.predict(dataTrain)
#    print("\nkernel SVM train ACC= ", acc(targetTrain, trainPredic))
    predicteSVM = clf4.predict(dataTest)
    ans5 = acc(targetTest, predicteSVM)
    SVMList.append(ans5)
#    print ("kernel SVM test ACC= ", ans5)
#%%
qdaACC = max(qdaList)
ldaACC = max(ldaList)
treeACC = max(treeList)
KNNACC = max(KNNList)
SVMACC = max(SVMList)
print('qda ACC', qdaACC)
print('lda ACC', ldaACC)
print('tree ACC', treeACC)
print('KNN ACC', KNNACC)
print('SVM ACC', SVMACC)
fig, ax = plt.subplots(3, 2, figsize=(10, 10))
fig.suptitle("SVM "+"test accuracy = " + str(SVMACC))
for a in ax.reshape(-1,1):
    a[0].set_xlabel("data")
    a[0].set_ylabel("class")
ax[0][0].plot(targetTest, color='red', label='Target test')
ax[0][0].plot(predicteSVM, color='blue', label='SVM predicted')
ax[0][0].legend()
ax[1][0].plot(targetTest, color='red', label='Target test')
ax[1][0].plot(predicteQDA, color='blue', label='QDA predicted')
ax[1][0].legend()
ax[0][1].plot(targetTest, color='red', label='Target test')  
ax[0][1].plot(predictKNN, color='blue', label='KNN predicted')
ax[0][1].legend()
ax[1][1].plot(targetTest, color='red', label='Target test')
ax[1][1].plot(predictLDA, color='blue', label='LDA predicted')
ax[1][1].legend()
ax[2][0].plot(targetTest, color='red', label='Targer test')
ax[2][0].plot(predictTree, color='blue', label='Tree predicted')
ax[2][0].legend()
plt.savefig('../Plots/MLACC.pdf')
#%%
fig, ax = plt.subplots(10, 2, figsize=(30, 100))
for i, j in 
    i[0].scatter( np.arange(dataTrain.shape[0]), dataTrain[:,0])
#for i in range(ax.shape[0]):
#    for j in range(ax.shape[1]):
#        ax[i][j].set_xlabel("dd")
#        ax[i][j].scatter( np.arange(dataTrain.shape[0]), dataTrain[:,0])
#    