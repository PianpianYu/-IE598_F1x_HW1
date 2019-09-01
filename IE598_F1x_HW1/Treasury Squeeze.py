#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:03:42 2019

@author: mac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/mac/Desktop/Machine Learning/module1/Treasury Squeeze test - DS1.csv', header = None)
#split data and target
np_df = np.array(df)
x, y = df .iloc[1:, :9], df.iloc[1:, 9]
np_x = np.array(x).astype(int)
np_y = np.array(y)
#get target names used later
np_y_names = np.array(y)
#turn string target to int velue
for i in range(len(np_y)): 
    if (np_y[i] == 'TRUE') : np_y[i] = 1
    else: np_y[i] = 0
np_y = np_y.astype(int)

#split the dataset into  atraining set and a testing set
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
x_train, x_test, y_train, y_test = train_test_split(np_x, np_y, test_size = 0.25, random_state = 33)
print(x_train.shape, y_train.shape)
#(675, 9) (675,)

#standardize the features
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#use SGDClassifier to get coefficient and linear model
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(x_train, y_train)
print(clf.coef_)
#[[-0.32102566 -0.32102566 -0.2739813   0.255889   -0.31049305 -0.79698101
#  -0.90557818 -0.30262563  0.88961499]]
print(clf.intercept_)
#[-0.16441189]
  
#predict one contract's squeeze result
print (clf.predict(scaler.transform([[1, 0, 0, 0, 1, 1, 0, 0, 1]]))) #7th
#[1] 
print (clf.predict(scaler.transform([[1, 0, 0, 0, 1, 1, 1, 0, 0]]))) #17th
#[0]
print (clf.decision_function(scaler.transform([[1, 0, 0, 0, 1, 1, 1, 0, 0]])))
#[-2.78369704]

#performance evaluate
from sklearn import metrics
y_train_pred = clf.predict(x_train)
print(metrics.accuracy_score(y_train, y_train_pred))
#0.5422222222222223
y_pred = clf.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))
#0.5288888888888889
print(metrics.classification_report(y_test, y_pred, target_names = ['TRUE', 'FALSE']))
#              precision    recall  f1-score   support

#        TRUE       0.62      0.59      0.60       136
#       FALSE       0.41      0.44      0.42        89

#    accuracy                           0.53       225
#   macro avg       0.51      0.51      0.51       225
#weighted avg       0.53      0.53      0.53       225
print(metrics.confusion_matrix(y_test, y_pred))
#[[80 56]
# [50 39]]
#----------- use only 2 features to plot ----------------#
#plot scatter
colors = ['green', 'orange']
for i in range(len(colors)):
    xs = x_train[:, 0][y_train == i]
    ys = x_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colors[i])
plt.legend(np_y_names)
plt.xlabel('price_crossing')
plt.ylabel('price_distortion')

#only 2 features used, here is 'price_crossing' & 'price_distortion'
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(np_x[:,0:2], np_y, test_size = 0.25, random_state = 33)
#standardize the features
scaler_2 = preprocessing.StandardScaler().fit(x_train_2)
x_train_2 = scaler_2.transform(x_train_2)
x_test_2 = scaler_2.transform(x_test_2)
#use SGDClassifier to get coefficient and linear model
clf_2 = SGDClassifier()
clf_2.fit(x_train_2, y_train_2)
print(clf_2.coef_)
#[[0.02223378 0.73689095]]
print(clf_2.intercept_)
#[-1.27778596]

#plot decision boundaries
x_min, x_max = x_train_2[:, 0].min() - .1, x_train_2[:, 0].max() + .1
y_min, y_max = x_train_2[:, 1].min() - .1, x_train_2[:, 1].max() + .1
xs = np.arange(x_min, x_max, 0.5)
fig, axes = plt.subplots(1, 1)
fig.set_size_inches(8, 8)
axes.set_xlabel('price_crossing')
axes.set_ylabel('price_distortion')
axes.set_xlim(x_min, x_max)
axes.set_ylim(y_min, y_max)
plt.sca(axes)
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c = y_train_2, cmap = plt.cm.prism)
ys = (-clf.intercept_ - xs * clf.coef_[0,0]) / clf.coef_[0,1]
plt.plot(xs, ys)

print("My name is Pianpian Yu")
print("My NetID is: py7")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
