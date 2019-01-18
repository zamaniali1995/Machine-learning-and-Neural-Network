#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:40:12 2019

@author: ali
"""
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score as acc
import pandas as pd
from sklearn.impute import SimpleImputer as imputer
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

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
        for i in range(1, 37):
            lable.append(data[data['ID']==i])
            lable[i-1]= imp.fit_transform(lable[i-1])
            if scale:
                scaler.fit_transform(lable[i-1][:,1:])
            
        data = np.concatenate(lable)
        data = shuffle(data)
        target, data = data[:, 0], data[:, 1:]
        np.savetxt('imputedTrain.csv', data, delimiter=',')
    else:
        data = imp.fit_transform(data)
        if scale:
            scaler.fit_transform(data[:, 1:])
        target, data = data[:, 0], data[:, 1:]
        np.savetxt('imputedTest.csv', data, delimiter=',')
    return data, dropIdx, target
# Class and feature number 
numFeatures = 19
numClass = 36
# Dataset directories
dirTestData = './TestDataSet/Test_Data.csv'
dirTestTarget = './TestDataSet/Test_Labels.csv'

dirwLayer1 = './Weights/wLayer1.csv'
dirwLayer2 = './Weights/wLayer2.csv'
dirwLayer3 = './Weights/wLayer3.csv'
dirwLayerO = './Weights/wLayerO.csv'
dirbLayer1 = './Weights/bLayer1.csv'
dirbLayer2 = './Weights/bLayer2.csv'
dirbLayer3 = './Weights/bLayer3.csv'
dirbLayerO = './Weights/bLayerO.csv'

# Loading  test data
dataTest, dropIdxTest, targetTest = dataLoader(0, 0, dirTestData,
                                        dirTestTarget, 1, 0, 0)
 
tmp = np.zeros((len(targetTest), numClass))
for i in range(len(targetTest)):
    for j in range(numClass):
        if targetTest[i]==j:
            tmp[i][j-1]=1
targetTestModifid = tmp
# loading weights
wLayer1 = pd.read_csv(dirwLayer1, header=None)
wLayer2 = pd.read_csv(dirwLayer2, header=None)
wLayer3 = pd.read_csv(dirwLayer3, header=None)
wLayerO = pd.read_csv(dirwLayerO, header=None)
bLayer1 = pd.read_csv(dirwLayer1, header=None)
bLayer2 = pd.read_csv(dirwLayer2, header=None)
bLayer3 = pd.read_csv(dirwLayer3, header=None)
bLayerO = pd.read_csv(dirwLayerO, header=None)

zLayer1 = np.nn.relu(tf.add(np.matmul(dataTest, wLayer1), bLayer1))
zLayer2 = np.nn.relu(tf.add(np.matmul(zLayer1, wLayer2), bLayer2))
zLayer3 = np.nn.relu(tf.add(np.matmul(zLayer2, wLayer3), bLayer3))
logit = np.add(tf.matmul(zLayer3, wLayerO), bLayerO)
yHat = tf.nn.sigmoid(tf.nn.relu(logit))
testPredicted = np.argmax(yHat, 1) + 1
print (acc(targetTest, testPredicted))