#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:11:34 2019

@author: mac
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 16:47:28 2019

@author: mac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

df = pd.read_csv('/Users/mac/Desktop/Machine Learning/module5/hw5_treasury yield curve data.csv', header = 0)
#EDA
#desribe the information of the dataset
summary = df.describe()
print(summary)
des= open('/Users/mac/Desktop/Machine Learning/module5/summary.csv', 'w')
print(df.describe(), file = des)
des.close()

#box-plot of varaibles
df = df.dropna(axis=0)
from pylab import boxplot
array = df.iloc[:,:].values
boxplot(array)
plt.xlabel("Variables Index")
plt.ylabel("Quartile Ranges")
plt.savefig('/Users/mac/Desktop/Machine Learning/module5/box-plot.jpg')
plt.show()

import seaborn as sns
sns.pairplot(df, height=2.5)
plt.tight_layout()
plt.savefig('/Users/mac/Desktop/Machine Learning/module5/pairplot.jpg')
plt.show()

#correlation matrix and heatmap
from pandas import DataFrame
corMat = DataFrame(df.corr())
#visualize correlations using heatmap
plt.figure(figsize=(21, 21))
sns.set(font_scale=1.5)
hm = sns.heatmap(corMat, cbar=True, annot=True, square=True,
                 fmt='.2f', annot_kws={'size': 15},
                 yticklabels=df.columns, xticklabels=df.columns)
plt.savefig('/Users/mac/Desktop/Machine Learning/module5/Heatmap.jpg')
plt.show()
sns.set(font_scale=1.0)
print(corMat)
Coefficient = open('/Users/mac/Desktop/Machine Learning/module5/Coefficient.csv', 'w')
print(corMat, file = Coefficient)
Coefficient.close()

#split
from sklearn.model_selection import train_test_split
X, y = df.iloc[:, 0:30],  df.iloc[:, 30]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state = 48)

# standardize the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#Part2 PCA
#calculate eigenvalue
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)

#calculate cumulative sum of explained variance
#for all components
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in           
           sorted(eigen_vals, reverse=True)]
print("expalined variance ration for all components is :", var_exp)
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1,31), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1,31), cum_var_exp, where='mid',
        label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.savefig('/Users/mac/Desktop/Machine Learning/module5/Explained Variance.jpg')
plt.show()

#set n_components = 3
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
cov_mat_3 = np.cov(X_train_pca.T)
eigen_vals_3, eigen_vecs_3 = np.linalg.eig(cov_mat_3)
print('\nEigenvalues \n%s' % eigen_vals_3)
tot = sum(eigen_vals_3)
var_exp_3 = [(i / tot) for i in           
           sorted(eigen_vals_3, reverse=True)]
print("expalined variance ration for 3 components is :", var_exp_3)
cum_var_exp_3 = np.cumsum(var_exp_3)
plt.bar(range(1,4), var_exp_3, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1,4), cum_var_exp_3, where='mid',
        label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.savefig('/Users/mac/Desktop/Machine Learning/module5/Explained Variance Transformed.jpg')
plt.show()
from sklearn.linear_model import LinearRegression
import time
lr_1 = LinearRegression()
start_time = time.time()
lr_1.fit(X_train_std, y_train)
end_time = time.time()
print('The baseline linear training time = {}'.format(end_time - start_time))
#Calculate its accuracy R2 score and RMSE for both in sample and out of sample (train and test sets)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
y_train_pred = lr_1.predict(X_train_std)
y_test_pred = lr_1.predict(X_test_std)
print('Baseline linear R^2 train: %.3f, test: %.3f' %
      (r2_score(y_train, y_train_pred),
       r2_score(y_test, y_test_pred)))
print('Baseline linear RMSE train: %.3f, test: %.3f' % (
       math.sqrt(mean_squared_error(y_train, y_train_pred)),
       math.sqrt(mean_squared_error(y_test, y_test_pred))))

#SVM Regresson model
from sklearn.svm import SVR
svm_1 = SVR(kernel = 'linear', C=1.0)
start_time = time.time()
svm_1.fit(X_train_std, y_train)
end_time = time.time()
print('The baseline SVR training time = {}'.format(end_time - start_time))
y_train_pred = svm_1.predict(X_train_std)
y_test_pred = svm_1.predict(X_test_std)
print('Baseline SVR R^2 train: %.3f, test: %.3f' %
      (r2_score(y_train, y_train_pred),
       r2_score(y_test, y_test_pred)))
print('Baseline SVR RMSE train: %.3f, test: %.3f' % (
       math.sqrt(mean_squared_error(y_train, y_train_pred)),
       math.sqrt(mean_squared_error(y_test, y_test_pred))))

#3 attributes
lr_3_1 = LinearRegression()
start_time = time.time()
lr_3_1.fit(X_train_pca, y_train)
end_time = time.time()
print('The transformed linear training time = {}'.format(end_time - start_time))
#Calculate its accuracy R2 score and RMSE for both in sample and out of sample (train and test sets)
y_train_pred = lr_3_1.predict(X_train_pca)
y_test_pred = lr_3_1.predict(X_test_pca)
print('Transformed linear R^2 train: %.3f, test: %.3f' %
      (r2_score(y_train, y_train_pred),
       r2_score(y_test, y_test_pred)))
print('Transformed linear RMSE train: %.3f, test: %.3f' % (
       math.sqrt(mean_squared_error(y_train, y_train_pred)),
       math.sqrt(mean_squared_error(y_test, y_test_pred))))

#SVM Regresson model
from sklearn.svm import SVR
svm_3_1 = SVR(kernel = 'linear', C=1.0)
start_time = time.time()
svm_3_1.fit(X_train_pca, y_train)
end_time = time.time()
print('The transformed SVR training time = {}'.format(end_time - start_time))
y_train_pred = svm_3_1.predict(X_train_pca)
y_test_pred = svm_3_1.predict(X_test_pca)
print('Transformed SVR R^2 train: %.3f, test: %.3f' %
      (r2_score(y_train, y_train_pred),
       r2_score(y_test, y_test_pred)))
print('Transformed SVR RMSE train: %.3f, test: %.3f' % (
       math.sqrt(mean_squared_error(y_train, y_train_pred)),
       math.sqrt(mean_squared_error(y_test, y_test_pred))))

print("My name is Pianpian Yu")
print("My NetID is: py7")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
