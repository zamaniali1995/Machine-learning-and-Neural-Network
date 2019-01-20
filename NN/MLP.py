#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:09:48 2019

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
    return data, dropIdx, target, lable
# Class and feature number 
numFeatures = 19
numClass = 36
# Dataset directories
dirTrain1 = '../DataSet/Train_DB_1.csv'
dirTrain2 = '../DataSet/Train_DB_2.csv'
dirTestData = '../DataSet/Test_Data.csv'
dirTestTarget = '../DataSet/Test_Labels.csv'
# Loading train and test data
dataTrain, dropIdxTrain, targetTrain, lable = dataLoader(dirTrain1, dirTrain2, dirTestData,
                                        dirTestTarget, 1, 1, 0)
#dataTest, dropIdxTest, targetTest = dataLoader(dirTrain1, dirTrain2, dirTestData,
#                                        dirTestTarget, 1, 0, 0)
 
tmp = np.zeros((len(targetTrain), numClass))
for i in range(len(targetTrain)):
    for j in range(numClass):
        if targetTrain[i]==j:
            tmp[i][j-1]=1
targetTrainModifid = tmp


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
    avg = np.mean(lable[i], axis=0)
    for j in range(dataTest.shape[0]):
        for k in range(dataTest.shape[1]):
            if nonDetected.iloc[j, k]:
                tmp.iloc[j, k] = avg[k + 1]
    dataTestModified.append(tmp)
dataTest = dataTestModified[0]
#%%
tmp = np.zeros((len(targetTest),numClass))
for i in range(len(targetTest)):
    for j in range(numClass):
        if targetTest[i]==j:
            tmp[i][j-1]=1
targetTestModifid = tmp
#%%
# set learning variables 
numHidden1 = 100
numHidden2 = 90
numHidden3 = 80
learnignRate = 0.0001
displayStep = 1
batchSize = 40
numEpoch = 1000
numCons1 = 0
numCons2 = 0
numCons3 = 0
numConsO = 0

tf.reset_default_graph()
# set some variables
# Placeholders
x = tf.placeholder(dtype=tf.float64, shape=(None, numFeatures), name='X')
yTrue = tf.placeholder(dtype=tf.float64, shape=(None, numClass), name='y')
# Variable
# Hidden layer 1
# Random init
wLayer1 = tf.Variable(tf.random_normal(shape=[numFeatures, numHidden1], stddev=0.1, dtype=tf.float64),
                      name='hlayerWeight1', dtype=tf.float64)
#wLayer1 = tf.get_variable(dtype=tf.float64, name='hlayerWeight1',
#                          initializer=tf.zeros(
#                                  shape=[numFeatures, numHidden1],
#                                  dtype=tf.float64))
                       
#bLayer1 = tf.get_variable(dtype=tf.float64, name='hlayerBias1', 
#                          initializer=tf.zeros(shape=
#                                               [numHidden1], dtype=tf.float64))
bLayer1 = tf.Variable(tf.random_normal(shape=[numHidden1], stddev=0.1, dtype=tf.float64),
                      name='hlayerBias1', dtype=tf.float64)

consLayer1 = tf.constant(numCons1, dtype=tf.float64)                                                                         
# Hidden layer 2
wLayer2 = tf.Variable(tf.random_normal(shape=[numHidden1, numHidden2], stddev=0.1, dtype=tf.float64),
                      name='hlayerWeight2', dtype=tf.float64)
# Zero init
#wLayer2 = tf.get_variable(dtype=tf.float64, name='hlayerWeight2',
#                          initializer=tf.zeros(
#                                  shape=[numHidden1, numHidden2],
#                                  dtype=tf.float64))
                       
#bLayer2 = tf.get_variable(dtype=tf.float64, name='hlayerBias2', 
#                          initializer=tf.zeros(shape=
#                                               [numHidden2], dtype=tf.float64))
bLayer2 = tf.Variable(tf.random_normal(shape=[numHidden2], stddev=0.1, dtype=tf.float64),
                      name='hlayerBias2', dtype=tf.float64)

consLayer2 = tf.constant(numCons2, dtype=tf.float64)                                                                         
# Hidden layer 3
wLayer3 = tf.Variable(tf.random_normal(shape=[numHidden2, numHidden3], stddev=0.1, dtype=tf.float64),
                      name='hlayerWeight3', dtype=tf.float64)
#wLayer3 = tf.get_variable(dtype=tf.float64, name='hlayerWeight3',
#                          initializer=tf.zeros(
#                                  shape=[numHidden2, numHidden3],
#                                  dtype=tf.float64))
#                       
#bLayer3 = tf.get_variable(dtype=tf.float64, name='hlayerBias3', 
#                          initializer=tf.zeros(shape=
#                                               [numHidden3], dtype=tf.float64))

bLayer3 = tf.Variable(tf.random_normal(shape=[numHidden3], stddev=0.1, dtype=tf.float64),
                      name='hlayerBias3', dtype=tf.float64)

consLayer3 = tf.constant(numCons3, dtype=tf.float64)                                                                         

# Output layer
wLayerOut = tf.Variable(tf.random_normal(shape=[numHidden3, numClass], stddev=0.1, dtype=tf.float64),
                      name='olayerWeight', dtype=tf.float64)

#wLayerOut = tf.get_variable(dtype=tf.float64, name='olayerWeight',initializer=tf.zeros
#                          (shape=[numHidden3, numClass], dtype=tf.float64))
bLayerOut = tf.Variable(tf.random_normal(shape=[numClass], stddev=0.1, dtype=tf.float64),
                      name='olayerBias', dtype=tf.float64)

#bLayerOut = tf.get_variable(dtype=tf.float64, name='olayerBias', initializer=tf.zeros(
#        shape=[numClass], dtype=tf.float64)) 
consLayerOut = tf.constant(numConsO, dtype=tf.float64)                                                                         
                                                                 
# Make the network
zLayer1 = tf.nn.relu(tf.add(tf.matmul(x, wLayer1), bLayer1) + consLayer1)
zLayer2 = tf.nn.relu(tf.add(tf.matmul(zLayer1, wLayer2), bLayer2) + consLayer2)
zLayer3 = tf.nn.relu(tf.add(tf.matmul(zLayer2, wLayer3), bLayer3) + consLayer3)
logit = tf.add(tf.matmul(zLayer3, wLayerOut), bLayerOut) + consLayerOut
yHat = tf.nn.sigmoid(tf.nn.relu(logit))
# Define loss function and optimizer 
crossEn = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=yTrue)
cost = tf.reduce_mean(crossEn)
optimizer = tf.train.AdamOptimizer(learning_rate=learnignRate).minimize(cost)
init = tf.initialize_all_variables()
# Session
trainAccList = []
testAccList = []
trainErrList = []
testErrList = []
with tf.Session() as sess:
    init.run()
    for epoch in range(numEpoch):
        trainLoss = 0
        for idx in range(len(dataTrain)//batchSize):
            InputList = {x: dataTrain[idx*batchSize:(idx+ 1)*batchSize],
                          yTrue: targetTrainModifid[idx*batchSize:(idx+ 1)*batchSize]}
            _, loss = sess.run([optimizer, cost], feed_dict=InputList)
            trainLoss += loss
        trainPredicted = sess.run(yHat, feed_dict={x: dataTrain})
        trainPredicted = np.argmax(trainPredicted, 1) + 1
        trainErrList.append(trainLoss)
        trainAccList.append(acc(targetTrain, trainPredicted))
        print("lass:", epoch, trainLoss)
        print("Train Acc", trainAccList[epoch])
        testPredicted = sess.run(yHat, feed_dict={x: dataTest})
        testPredicted = np.argmax(testPredicted, 1) + 1
        testErrList.append(sess.run(cost, feed_dict={x: dataTest, yTrue: targetTestModifid}))
        testAccList.append(acc(targetTest, testPredicted))
        print("Test Acc", testAccList[epoch])
    w1, b1, w2, b2, w3, b3, wO, bO = sess.run([wLayer1, bLayer1, wLayer2,
                                               bLayer2, wLayer3, bLayer3, wLayerOut, bLayerOut])
        
    np.savetxt('./TestModel/Weights/wLayer1.csv', w1, delimiter=',')
    np.savetxt('./TestModel/Weights/bLayer1.csv', b1, delimiter=',')
    np.savetxt('./TestModel/Weights/wLayer2.csv', w2, delimiter=',')
    np.savetxt('./TestModel/Weights/bLayer2.csv', b2, delimiter=',')
    np.savetxt('./TestModel/Weights/wLayer3.csv', w3, delimiter=',')
    np.savetxt('./TestModel/Weights/bLayer3.csv', b3, delimiter=',')
    np.savetxt('./TestModel/Weights/wLayerO.csv', wO, delimiter=',')
    np.savetxt('./TestModel/Weights/bLayerO.csv', bO, delimiter=',')
#%%
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle("test accuracy = " + str(testAccList[numEpoch-1]))
for a in ax.reshape(-1,1):
    a[0].set_xlabel("epochs")
ax[0][0].plot(trainErrList[:500], color='red', label='train loss')
ax[0][0].plot(testErrList[:500], color='blue', label='test loss')
ax[0][0].legend()
ax[1][0].plot(trainErrList, color='red', label='train loss')
ax[1][0].plot(testErrList, color='blue', label='test loss')
ax[1][0].legend()
ax[0][1].plot(trainAccList[:500], color='red', label='train accuracy')  
ax[0][1].plot(testAccList[:500], color='blue', label='test accuracy')
ax[0][1].legend()
ax[1][1].plot(trainAccList, color='red', label='train accuracy')
ax[1][1].plot(testAccList, color='blue', label='test accuracy')
ax[1][1].legend()
plt.savefig("../Plots/results"+".pdf")