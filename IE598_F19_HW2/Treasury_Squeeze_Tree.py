#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

#import table information
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
 
#use DecisionTreeClassifier to get predict model
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 4, random_state = 1)
tree.fit(x_train, y_train)

print (tree.predict([[1, 1, 1, 0, 0, 1,	1, 0, 1]]))
#[1] 
print (tree.predict([[1, 0, 0, 0, 1, 1, 1, 0, 0]]))
#[0]

#performance evaluate
from sklearn import metrics
y_train_pred = tree.predict(x_train)
print( metrics.accuracy_score(y_train, y_train_pred) )
#0.674074074074074

y_pred = tree.predict(x_test)
print( metrics.accuracy_score(y_test, y_pred) )
#0.6844444444444444

#plot decision regions with first 2 features
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(np_x[:,0:2], np_y, test_size = 0.25, random_state = 33)
tree_2 = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state = 1)
tree_2.fit(x_train_2, y_train_2)
plot_decision_regions(np_x[:, 0:2], np_y, clf=tree_2)
plt.xlabel('price_crossing')
plt.ylabel('price_distortion')
plt.legend(loc='upper left')
plt.show()
#analize the performance under different depth
d_range = range(1,16)
scores_train = []
scores_test = []
for d in d_range:
    tree = DecisionTreeClassifier(criterion = 'gini', max_depth = d, random_state = 1)
    tree.fit(x_train, y_train)
    y_train_pred = tree.predict(x_train)
    y_pred = tree.predict(x_test)
    scores_train.append(metrics.accuracy_score(y_train, y_train_pred))
    #Compute accuracy on the testing set
    scores_test.append(metrics.accuracy_score(y_test, y_pred)) 
    print(d, metrics.accuracy_score(y_train, y_train_pred), metrics.accuracy_score(y_test, y_pred))
#1 0.6281481481481481 0.6488888888888888
#2 0.6444444444444445 0.6666666666666666
#3 0.6548148148148148 0.6933333333333334
#4 0.674074074074074 0.6844444444444444
#5 0.6977777777777778 0.6444444444444445
#6 0.7437037037037038 0.64
#7 0.7837037037037037 0.6488888888888888
#8 0.8311111111111111 0.5688888888888889
#9 0.8533333333333334 0.5822222222222222
#10 0.8533333333333334 0.5866666666666667
#11 0.8533333333333334 0.5866666666666667
#12 0.8533333333333334 0.5866666666666667
#13 0.8533333333333334 0.5866666666666667
#14 0.8533333333333334 0.5866666666666667
#15 0.8533333333333334 0.5866666666666667    
#plot Model Complexicity and Over/Underfitting
plt.figure()
plt.title('DTs: Varying Number of Depth')
plt.plot(d_range, scores_train, label = 'Training Accuracy')
plt.plot(d_range, scores_test, label = 'Testing Accuracy')
plt.legend()
plt.xlabel('Number of Depth')
plt.ylabel('Accuracy')
plt.show()

#use RandomForestClassifier to reduce overfitting in the desicion tree model because of so many features
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini',
                                max_depth = 4,
                                n_estimators=25,
                                random_state=1,
                                n_jobs=2)         #default 9^(1/2) features
forest.fit(x_train, y_train)
print (forest.predict([[1, 0, 0, 0, 1, 1, 0, 0, 1]]))
#[1] 
print (forest.predict([[1, 0, 0, 0, 1, 1, 1, 0, 0]]))
#[0]

y_train_pred_forest = forest.predict(x_train)
y_pred_forest = forest.predict(x_test)
print(metrics.accuracy_score(y_train, y_train_pred_forest), metrics.accuracy_score(y_test, y_pred_forest))
#0.6696296296296296 0.6888888888888889   

print("My name is Pianpian Yu")
print("My NetID is: py7")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
