#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:24:23 2019

@author: mac
"""
import pandas as pd
import numpy as np
import time

df = pd.read_csv('/Users/mac/Desktop/Machine Learning/module6/ccdefault.csv', header = 0)

#Part 1: Random test train splits
X = df.iloc[:,0:23]
y = df.iloc[:,23]
train_acc = np.ndarray(10)
test_acc = np.ndarray(10)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 4, random_state = 1)
start_time = time.time()
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state = i+1)
    tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred= tree.predict(X_test)
    train_acc[i]=metrics.accuracy_score(y_train,y_train_pred)
    test_acc[i]=metrics.accuracy_score(y_test,y_test_pred)
end_time = time.time()
print('The split time = {}'.format(end_time - start_time))
print('Individula score for in-sample: %s' %train_acc)
print('Individula score for out-of-sample: %s' %test_acc)
print('Mean score for in-sample: %.3f' %train_acc.mean())
print('STD score for in-sample: %.3f' %train_acc.std())
print('Mean score for out-of-sample: %.3f' %test_acc.mean())
print('STD score for out-of-sample: %.3f' %test_acc.std())

#Part 2: Cross validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
#from sklearn.pipeline import make_pipeline
dt = DecisionTreeClassifier(criterion = 'gini', max_depth = 4,random_state = 1)

#from the first part, the optimal random_state is 3 for out-of-samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state = 3)
start_time = time.time()
scores = cross_val_score(estimator=dt, X=X_train, y=y_train, cv=10, n_jobs=-1)
end_time = time.time()
print('The evaluation of cross validation time = {}'.format(end_time - start_time))
print('CV accuracy scores: %s' % scores)
print('CV accuracy mean score: %.3f' %scores.mean())
print('CV accuracy std score: %.3f' %scores.std())

#Run the out-of-sample accuracy score
#use GridSearchCV to fine optimal max_depth
parameters = {'max_depth':[1,10]}
dt = DecisionTreeClassifier(criterion = 'gini',random_state = 1)
start_time = time.time()
cv = GridSearchCV(dt, param_grid=parameters, scoring='accuracy', cv =10)
scores = cross_val_score(cv, X_train, y_train,
                          scoring='accuracy', cv=10)
cv.fit(X_train, y_train)
end_time = time.time()
print('The GridSearchCV training time = {}'.format(end_time - start_time))
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('CV score when tuning hyperparameter:', cv.score(X_test, y_test))
print('CV best parameter:', cv.best_params_)

#retrain after tuning hyperparameter
dt = DecisionTreeClassifier(criterion = 'gini', max_depth = 1, random_state = 1)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print('Accuracy score for out-of-sample: %.6f' %metrics.accuracy_score(y_test, y_pred))

#a different performance metrix-confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
print(classification_report(y_test, y_pred))
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

print('My name is Pianpian Yu')
print('My NetID is: py7')
print('I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.')
