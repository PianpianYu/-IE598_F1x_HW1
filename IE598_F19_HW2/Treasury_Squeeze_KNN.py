#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv('/Users/mac/Desktop/Machine Learning/module2/Treasury Squeeze test - DS1.csv', header = None)
#split data and target
np_df = np.array(df)
x, y = df .iloc[1:, :9], df.iloc[1:, 9]
np_x = np.array(x).astype(int)
np_y = np.array(y)
#turn string target to int velue
for i in range(len(np_y)): 
    if (np_y[i] == 'TRUE') : np_y[i] = 1
    else: np_y[i] = 0
np_y = np_y.astype(int)

#split the dataset into a training set and a testing set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(np_x, np_y, test_size = 0.25, random_state = 33)
print(x_train.shape, y_train.shape)
#(675, 9) (675,)

#use KNeighborsClassifier to get predict model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
knn.fit(x_train, y_train)

#performance evaluate
print (knn.predict([[1, 0, 0, 0, 1, 1, 0, 0, 1]]))
#[1] 
print (knn.predict([[1, 0, 0, 0, 1, 1, 1, 0, 0]]))
#[0]

from sklearn import metrics
y_train_pred = knn.predict(x_train)
print( metrics.accuracy_score(y_train, y_train_pred) )
#0.7437037037037038

y_pred = knn.predict(x_test)
print( metrics.accuracy_score(y_test, y_pred) )
#0.6355555555555555

#plot decision regions with first 2 features
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(np_x[:, 0:2], np_y, test_size = 0.25, random_state = 33)
knn_2 = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
knn_2.fit(x_train_2, y_train_2)
#find typos here: classifier->clf, test_idx: no such parameter,but feature_index
plot_decision_regions(np_x[:, 0:2], np_y, clf=knn_2)
plt.xlabel('price_crossing')
plt.ylabel('price_distortion')
plt.legend(loc='upper left')
plt.show()

#analize the performance under different k
k_range = range(1,26)
scores_train = []
scores_test = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k, p = 2, metric = 'minkowski')
    knn.fit(x_train, y_train)
    y_train_pred = knn.predict(x_train)
    y_pred = knn.predict(x_test)    
    scores_train.append(metrics.accuracy_score(y_train, y_train_pred))  
    scores_test.append(metrics.accuracy_score(y_test, y_pred))
    print(k, metrics.accuracy_score(y_train, y_train_pred), metrics.accuracy_score(y_test, y_pred))
#1 0.8251851851851851 0.5777777777777777
#2 0.7762962962962963 0.64
#3 0.7881481481481482 0.6177777777777778
#4 0.7362962962962963 0.6177777777777778
#5 0.7437037037037038 0.6355555555555555
#6 0.717037037037037 0.6533333333333333
#7 0.7348148148148148 0.6311111111111111
#8 0.7140740740740741 0.6266666666666667
#9 0.7229629629629629 0.6044444444444445
#10 0.6948148148148148 0.6311111111111111
#11 0.6933333333333334 0.6222222222222222
#12 0.6918518518518518 0.6311111111111111
#13 0.6962962962962963 0.6444444444444445
#14 0.6933333333333334 0.6666666666666666
#15 0.7007407407407408 0.64
#16 0.6948148148148148 0.6711111111111111
#17 0.6962962962962963 0.6711111111111111
#18 0.6888888888888889 0.6888888888888889
#19 0.6948148148148148 0.6622222222222223
#20 0.68 0.6844444444444444
#21 0.6903703703703704 0.6711111111111111
#22 0.6948148148148148 0.6711111111111111
#23 0.6814814814814815 0.6577777777777778
#24 0.6814814814814815 0.6755555555555556
#25 0.6755555555555556 0.6933333333333334
    
#plot Model Complexicity and Over/Underfitting    
plt.figure()
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k_range, scores_train, label = 'Training Accuracy')
plt.plot(k_range, scores_test, label = 'Testing Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

print("My name is Pianpian Yu")
print("My NetID is: py7")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


    
    
