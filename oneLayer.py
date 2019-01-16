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
import random
######################## set learning variables ##################
numFeatures = 19
numHidden1 = 5
numHidden2 = 20
numClass = 36
learnignRate = 0.1
numEpoch = 300
displayStep = 1
batchSize = 100

def dataLoader(dir_dtrain_1, dir_dtrain_2, dir_data_test, dir_lable_test,
               merge_flag, is_train):
    if is_train:
        if merge_flag:
            data_set1 = pd.read_csv(dir_dtrain_1)
            data_set2 = pd.read_csv(dir_dtrain_2)
            data_set = pd.concat([data_set1, data_set2])
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
        for i in range(1, 37):
            lable.append(data[data['ID']==i])
            lable[i-1]= imp.fit_transform(lable[i-1])
#            scaler.fit_transform(lable[i-1][:,1:])
            
        data= np.concatenate(lable)
        target, data = data[:, 0], data[:, 1:]
        np.savetxt('imputedTrain.csv', data, delimiter=',')
#        np.savetxt('target.csv', target, delimiter=',')
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
dataTrain, dropIdxTrain, targetTrain = dataLoader(train_dir_1, train_dir_2, test_data_dir,
                                        test_target_dir, 1, 1)
dataTest, dropTidxTest, targetTest = dataLoader(train_dir_1, train_dir_2, test_data_dir,
                                      test_target_dir, 1, 0) 
tmp = np.zeros((len(targetTrain),numClass))
for i in range(len(targetTrain)):
    for j in range(numClass):
        if targetTrain[i]==j:
            tmp[i][j-1]=1
targetTrain = tmp

tmp = np.zeros((len(targetTest),numClass))
for i in range(len(targetTest)):
    for j in range(numClass):
        if targetTest[i]==j:
            tmp[i][j-1]=1
targetTest1 =   targetTest
targetTest = tmp
#%%
#dataTrain = pd.read_csv('imputedTrain.csv')
#targetTrain = pd.read_csv('target.csv')
#targetTrain, dataTrain = data_set.iloc[:, 0], data_set.iloc[:, 1:]
#dataTrain = np.matrix.transpose(dataTrain[0])
tf.reset_default_graph()
######################## set some variables #######################
# Placeholder
x = tf.placeholder(dtype=tf.float32, shape=(None, numFeatures), name='X')
yTrue = tf.placeholder(dtype=tf.float32, shape=(None, numClass), name='y')
# Variable
# Hidden layer 1
wLayer1 = tf.get_variable(dtype=tf.float32, name='hlayerWeight1',
                          initializer=tf.zeros(
                                  shape=[numFeatures, numHidden1],
                                  dtype=tf.float32))
                       
bLayer1 = tf.get_variable(dtype=tf.float32, name='hlayerBias1', 
                          initializer=tf.zeros(shape=
                                               [numHidden1], dtype=tf.float32))
# Hidden layer 2
wLayer2 = tf.get_variable(dtype=tf.float32, name='hlayerWeight2',
                          initializer=tf.zeros(
                                  shape=[numHidden1, numHidden2],
                                  dtype=tf.float32))
                       
bLayer2 = tf.get_variable(dtype=tf.float32, name='hlayerBias2', 
                          initializer=tf.zeros(shape=
                                               [numHidden2], dtype=tf.float32))
                                                                         
# Output layer
wLayer3 = tf.get_variable(dtype=tf.float32, name='olayerWeight',initializer=tf.zeros
                          (shape=[numHidden2, numClass], dtype=tf.float32))
bLayer3 = tf.get_variable(dtype=tf.float32, name='olayerBias', initializer=tf.zeros(
        shape=[numClass], dtype=tf.float32)) 
                                                                        
# Make the network
zLayer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, wLayer1), bLayer1))
zLayer2 = tf.nn.sigmoid(tf.add(tf.matmul(zLayer1, wLayer2), bLayer2))
logit = tf.add(tf.matmul(zLayer2, wLayer3), bLayer3)
yHat = tf.sigmoid(logit)
#zeros = tf.zeros_like(targetTest)
#yPredicted = zeros[np.arange(len(targetTest)), yHat.armax(1)] = 1 
#yHat = tf.nn.sigmoid(z)
# 
crossEn = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=yTrue)
cost = tf.reduce_mean(crossEn)
#
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learnignRate).minimize(cost)
init = tf.initialize_all_variables()
#a = dataTrain[0].reshape(-1, 1)
#a = np.matrix.transpose(a)
#targetTrain = targetTrain.reshape(-1, 1)
with tf.Session() as sess:
    init.run()
    for epoch in range(numEpoch):
        trainLoss = 0
#        for idx in range(len(dataTrain)//batchSize):
        idx = random.randint(0,len((dataTrain)//batchSize))
        InputList = {x: dataTrain[idx*batchSize:(idx+1)*batchSize],
                      yTrue: targetTrain[idx*batchSize:(idx+1)*batchSize]}
        _, loss, predictedTrain = sess.run([optimizer, cost, yHat], feed_dict=InputList)
        trainLoss += loss
        print(epoch, trainLoss)
    predicted = sess.run(yHat, feed_dict={x: dataTest})
#    correctPrediction = tf.equal(tf.argmax(yHat, 1), tf.argmax(yTrue, 1))
#    accuracy = tf.reduce_mean(tf.case(correctPrediction, "float32"))
    predicted1 = np.argmax(predicted, 1) + 1
    print(acc(targetTest1, predicted1))
#    print(acc(tf.argmax(targetTest), tf.argmax(sess.run(yHat, feed_dict={x: dataTest})), 1))
#    print("Accuracy:", accuracy.eval({x: dataTest, yTrue: targetTest}))