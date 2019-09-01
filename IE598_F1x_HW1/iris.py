#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:02:24 2019

@author: mac
"""

#My first machine learning model
import sklearn
print( 'The scikit learn version is {}.'.format(sklearn.__version__))

from sklearn import datasets
iris = datasets.load_iris() 
x_iris, y_iris = iris.data, iris.target 
print (x_iris.shape, y_iris.shape)
#(150, 4) (150,)
print (x_iris[0], y_iris[0])
#[5.1 3.5 1.4 0.2] 0

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#get dataset with only the first two arrtibutes
x, y = x_iris[:, :2], y_iris
#split the dataset into a training and a testing set
#test set will be the 25% taken randomly
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 33)
print(x_train.shape, y_train.shape)
#(112, 2) (112,)
#standardize the features
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
 
import matplotlib.pyplot as plt
colors = ['red', 'greenyellow', 'blue']
for i in range(len(colors)):
    xs = x_train[:, 0][y_train == i]
    ys = x_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(x_train, y_train)
print(clf.coef_)
#[[-30.18760182  16.74272191]
# [ -0.4645969   -6.08587279]
# [  7.25962685  -8.21344965]]
print(clf.intercept_)
#[-18.38846339  -6.64262377 -10.37455866]

import numpy as np
x_min, x_max = x_train[:, 0].min() - .5, x_train[:, 0].max() + .5
y_min, y_max = x_train[:, 1].min() - .5, x_train[:, 1].max() + .5
xs = np.arange(x_min, x_max, 0.5)
fig, axes = plt.subplots(1, 3)
fig.set_size_inches(10, 6)
for i in [0, 1, 2]:
    axes[i].set_aspect('equal')
    axes[i].set_title('Class' + str(i) + ' versus the rest')
    axes[i].set_xlabel('Sepel length')
    axes[i].set_ylabel('Sepel width')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    plt.sca(axes[i])
    plt.scatter(x_train[:, 0], x_train[:, 1], c = y_train, cmap = plt.cm.prism)
    ys = (-clf.intercept_[i] - xs * clf.coef_[i, 0]) / clf.coef_[i, 1]
    plt.plot(xs, ys)

print (clf.predict(scaler.transform([[4.7, 3.1]])))
#[0]
print (clf.decision_function(scaler.transform([[4.7, 3.1]])))
#[[ 21.047417    -6.57164387 -20.21200365]] 
from sklearn import metrics
y_train_pred = clf.predict(x_train)
print(metrics.accuracy_score(y_train, y_train_pred))
#0.8214285714285714 
y_pred = clf.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))
#0.7105263157894737 
print(metrics.classification_report(y_test, y_pred, target_names=iris.target_names))
#              precision    recall  f1-score   support

#      setosa       1.00      1.00      1.00         8
#  versicolor       0.50      0.55      0.52        11
#   virginica       0.72      0.68      0.70        19

#    accuracy                           0.71        38
#   macro avg       0.74      0.74      0.74        38
#weighted avg       0.72      0.71      0.71        38
print(metrics.confusion_matrix(y_test, y_pred))
#[[ 8  0  0]
# [ 0  6  5]
# [ 0  6 13]]

print("My name is Pianpian Yu")
print("My NetID is: py7")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
 
