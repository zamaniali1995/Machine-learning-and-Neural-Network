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
#import random
from sklearn.utils import shuffle

def dataLoader(dir_dtrain_1, dir_dtrain_2, dir_data_test, dir_lable_test,
               merge_flag, is_train):
    dataSet = 0
    if is_train:
        if merge_flag:
            dataSet1 = pd.read_csv(dir_dtrain_1)
            dataSet2 = pd.read_csv(dir_dtrain_2)
            dataSet = pd.concat([dataSet1, dataSet2])
#            dataSet = shuffle(dataSet)
        else:
            dataSet = pd.read_csv(dir_dtrain_1)
#            dataSet = shuffle(dataSet)
    else:
         data = pd.read_csv(dir_data_test)
         target = pd.read_csv(dir_lable_test)
         dataSet = pd.concat([target, data], axis=1)
         
    dataSet.columns=['ID', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 
                      'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 
                      'F17', 'F18', 'F19']
    lable=[]
    scaler = StandardScaler(copy=False)
    dataSet = dataSet.replace(0, np.NaN)
    imp = imputer(missing_values=np.NaN, strategy='mean')
# Drops rows with all nans
    drop_idx=[]
    nan_detected_data_set = pd.isnull(dataSet)
    for i in range(dataSet.shape[0]):
        cnt = 0
        for j in range(1, dataSet.shape[1]):
            if nan_detected_data_set.iloc[i, j]:
                cnt += 1
        if cnt==dataSet.shape[1]-1:
            drop_idx.append(i)
    data = dataSet.drop(dataSet.index[drop_idx])
    if is_train:
# Replaces nan features with mean and scales features
        for i in range(1, 37):
            lable.append(data[data['ID']==i])
            lable[i-1]= imp.fit_transform(lable[i-1])
            scaler.fit_transform(lable[i-1][:,1:])
            
        data = np.concatenate(lable)
        data = shuffle(data)
#        data = np.round(data)
        target, data = data[:, 0], data[:, 1:]
        np.savetxt('imputedTrain.csv', data, delimiter=',')
#        np.savetxt('target.csv', target, delimiter=',')
    else:
        data = imp.fit_transform(data)
        scaler.fit_transform(data[:, 1:])
        target, data = data[:, 0], data[:, 1:]
        np.savetxt('imputedTest.csv', data, delimiter=',')
    return data, drop_idx, target
######################## Load dataset ############################
train_dir_1= './DataSet/Train_DB_1.csv'
train_dir_2 = './DataSet/Train_DB_2.csv'
test_data_dir = './DataSet/Test_Data.csv'
test_target_dir = './DataSet/Test_Labels.csv'

numFeatures = 19
numClass = 36

dataTrain, dropIdxTrain, targetTrain = dataLoader(train_dir_1, train_dir_2, test_data_dir,
                                        test_target_dir, 0, 1)
dataTest, dropTidxTest, targetTest = dataLoader(train_dir_1, train_dir_2, test_data_dir,
                                      test_target_dir, 1, 0) 
tmp = np.zeros((len(targetTrain), numClass))
for i in range(len(targetTrain)):
    for j in range(numClass):
        if targetTrain[i]==j:
            tmp[i][j-1]=1
targetTrainModifid = tmp

tmp = np.zeros((len(targetTest),numClass))
for i in range(len(targetTest)):
    for j in range(numClass):
        if targetTest[i]==j:
            tmp[i][j-1]=1
#targetTest1 =   targetTest
targetTestModifid = tmp
#%%
######################## set learning variables ##################

numHidden1 = 50
numHidden2 = 20
numHidden3 = 30
learnignRate = 0.001
displayStep = 1
batchSize = 20
numEpoch = 10000


#dataTrain = pd.read_csv('imputedTrain.csv')
#targetTrain = pd.read_csv('target.csv')
#targetTrain, dataTrain = data_set.iloc[:, 0], data_set.iloc[:, 1:]
#dataTrain = np.matrix.transpose(dataTrain[0])
tf.reset_default_graph()
######################## set some variables #######################
# Placeholder
x = tf.placeholder(dtype=tf.float64, shape=(None, numFeatures), name='X')
yTrue = tf.placeholder(dtype=tf.float64, shape=(None, numClass), name='y')
# Variable
# Hidden layer 1
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

consLayer1 = tf.constant(1, dtype=tf.float64)                                                                         
# Hidden layer 2
wLayer2 = tf.Variable(tf.random_normal(shape=[numHidden1, numHidden2], stddev=0.1, dtype=tf.float64),
                      name='hlayerWeight2', dtype=tf.float64)
#wLayer2 = tf.get_variable(dtype=tf.float64, name='hlayerWeight2',
#                          initializer=tf.zeros(
#                                  shape=[numHidden1, numHidden2],
#                                  dtype=tf.float64))
                       
#bLayer2 = tf.get_variable(dtype=tf.float64, name='hlayerBias2', 
#                          initializer=tf.zeros(shape=
#                                               [numHidden2], dtype=tf.float64))
bLayer2 = tf.Variable(tf.random_normal(shape=[numHidden2], stddev=0.1, dtype=tf.float64),
                      name='hlayerBias2', dtype=tf.float64)

consLayer2 = tf.constant(1, dtype=tf.float64)                                                                         
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

consLayer3 = tf.constant(1, dtype=tf.float64)                                                                         

# Output layer
wLayerOut = tf.Variable(tf.random_normal(shape=[numHidden3, numClass], stddev=0.1, dtype=tf.float64),
                      name='olayerWeight', dtype=tf.float64)

#wLayerOut = tf.get_variable(dtype=tf.float64, name='olayerWeight',initializer=tf.zeros
#                          (shape=[numHidden3, numClass], dtype=tf.float64))
bLayerOut = tf.Variable(tf.random_normal(shape=[numClass], stddev=0.1, dtype=tf.float64),
                      name='olayerBias', dtype=tf.float64)

#bLayerOut = tf.get_variable(dtype=tf.float64, name='olayerBias', initializer=tf.zeros(
#        shape=[numClass], dtype=tf.float64)) 
consLayerOut = tf.constant(1, dtype=tf.float64)                                                                         
                                                                 
# Make the network
zLayer1 = tf.nn.relu(tf.add(tf.matmul(x, wLayer1), bLayer1) + consLayer1)
zLayer2 = tf.nn.relu(tf.add(tf.matmul(zLayer1, wLayer2), bLayer2) + consLayer2)
zLayer3 = tf.nn.relu(tf.add(tf.matmul(zLayer2, wLayer3), bLayer3) + consLayer3)
logit = tf.add(tf.matmul(zLayer3, wLayerOut), bLayerOut,) + consLayerOut
yHat = tf.nn.sigmoid(tf.nn.relu(logit))

#zeros = tf.zeros_like(targetTest)
#yPredicted = zeros[np.arange(len(targetTest)), yHat.armax(1)] = 1 
#yHat = tf.nn.sigmoid(z)
# 
crossEn = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=yTrue)
cost = tf.reduce_mean(crossEn)
#
optimizer = tf.train.AdamOptimizer(learning_rate=learnignRate).minimize(cost)
init = tf.initialize_all_variables()
#a = dataTrain[0].reshape(-1, 1)
#a = np.matrix.transpose(a)
#targetTrain = targetTrain.reshape(-1, 1)
with tf.Session() as sess:
    init.run()
    for epoch in range(numEpoch):
        trainLoss = 0
        for idx in range(len(dataTrain)//batchSize):
#        idx = random.randint(0,len((dataTrain)//batchSize))
            InputList = {x: dataTrain[idx*batchSize:(idx+ 1)*batchSize],
                          yTrue: targetTrainModifid[idx*batchSize:(idx+ 1)*batchSize]}
            _, loss = sess.run([optimizer, cost], feed_dict=InputList)
            trainLoss += loss
        trainPredicted = sess.run(yHat, feed_dict={x: dataTrain})
        trainPredicted = np.argmax(trainPredicted, 1) + 1
        print("lass:", epoch, trainLoss)
        print("Train Acc", acc(targetTrain, trainPredicted))
#    trainPredicted = sess.run(yHat, feed_dict={x: dataTrain})
    testPredicted = sess.run(yHat, feed_dict={x: dataTest})
#    correctPrediction = tf.equal(tf.argmax(yHat, 1), tf.argmax(yTrue, 1))
#    accuracy = tf.reduce_mean(tf.case(correctPrediction, "float64"))
    testPredicted = np.argmax(testPredicted, 1) + 1
#    trainPredicted1 = np.argmax(trainPredicted, 1) + 1
    print(acc(targetTest, testPredicted))
#    print(acc(targetTrain, trainPredicted1)) 
#    print(acc(tf.argmax(targetTest), tf.argmax(sess.run(yHat, feed_dict={x: dataTest})), 1))
#    print("Accuracy:", accuracy.eval({x: dataTest, yTrue: targetTest}))