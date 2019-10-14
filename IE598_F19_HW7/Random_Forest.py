#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:51:55 2019

@author: mac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

df = pd.read_csv('/Users/mac/Desktop/Machine Learning/module7/ccdefault.csv', header = 0)

#Part 1: Random forest estimators
X = df.iloc[:,0:23]
y = df.iloc[:,23]
X= np.array(X)
y = np.array(y)
#split training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state = 3)
from sklearn.ensemble import RandomForestClassifier
#find optimal n_estimators
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
#parameters = {'n_estimators': [5,10,15,20,25,30,35,40,45,50,60,65,70]}
estimator_list = [5,10,15,20,25,30,35,40,45,50,60,65,70]
#estimator_list = [5,10]
scores = []
times = []
for i in estimator_list:
    forest = RandomForestClassifier(criterion='gini', n_estimators=i, random_state=1, n_jobs=2)
    start_time = time.time()
    forest.fit(X_train, y_train)
    end_time = time.time()
    times.append(end_time - start_time)
    scores.append(np.mean(cross_val_score(forest, X_train, y_train, scoring='roc_auc', cv=10)))
#print('computation times are: %.6f' %times)
print(times)
#print('accuracy for im-samples are: %.6f' %scores)
print(scores)

#Part2: Random forest feature importance
#select n_estimator = 70
feat_labels = df.columns[0:23]
forest = RandomForestClassifier(criterion='gini', n_estimators=70, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
feat_labels_new = []
for f in range(X_train.shape[1]):
    feat_labels_new.append(feat_labels[indices[f]])
    print("%2d %-*s %f" %(f+1, 30, feat_labels[indices[f]],importances[indices[f]]))
#plot different features ranks
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]),feat_labels_new, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.savefig('/Users/mac/Desktop/Machine Learning/module7/feature_importance.jpg')
plt.show()

print('My name is Pianpian Yu')
print('My NetID is: py7')
print('I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.')
