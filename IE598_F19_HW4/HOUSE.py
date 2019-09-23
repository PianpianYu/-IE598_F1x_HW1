#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 19:51:33 2019

@author: mac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('/Users/mac/Desktop/Machine Learning/module4/housing.xlsx', header = 0)
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
#EDA
summary = df.describe()
print(summary)
des = open('/Users/mac/Desktop/Machine Learning/module4/summary.csv', 'w')
print(df.describe(), file = des)
des.close()
#box-plot of 13 attributes
from pylab import boxplot
array = df.iloc[:,0:13].values
boxplot(array)
plt.xlabel("Attribute Index")
plt.ylabel(("Quartile Ranges"))
plt.savefig('/Users/mac/Desktop/Machine Learning/module4/box-plot.jpg')
plt.show()

df = df.dropna(axis=0)
import seaborn as sns
sns.pairplot(df, size=2.5)
plt.tight_layout()
plt.savefig('/Users/mac/Desktop/Machine Learning/module4/pairplot.jpg')
plt.show()

#13x13 correlation matrix and heatmap
from pandas import DataFrame
corMat = DataFrame(df.corr())
#visualize correlations using heatmap
plt.figure(figsize=(12, 12))
sns.set(font_scale=1.5)
hm = sns.heatmap(corMat, cbar=True, annot=True, square=True,
                 fmt='.2f', annot_kws={'size': 15},
                 yticklabels=df.columns, xticklabels=df.columns)
plt.savefig('/Users/mac/Desktop/Machine Learning/module4/Heatmap.jpg')
plt.show()

print(corMat)
Coefficient = open('/Users/mac/Desktop/Machine Learning/module4/Coefficient.csv', 'w')
print(corMat, file = Coefficient)
Coefficient.close()

#standardize the variables for better convergence of the GD algorithm
class LinearRegressionGD(object):
       def __init__(self, eta=0.001, n_iter=20):
           self.eta = eta
           self.n_iter = n_iter
       def fit(self, X, y):
           self.w_ = np.zeros(1 + X.shape[1])
           self.cost_ = []
           for i in range(self.n_iter):
               output = self.net_input(X)
               errors = (y - output)
               self.w_[1:] += self.eta * X.T.dot(errors)
               self.w_[0] += self.eta * errors.sum()
               cost = (errors**2).sum() / 2.0
               self.cost_.append(cost)
           return self
       def net_input(self, X):
           return np.dot(X, self.w_[1:]) + self.w_[0]
       def predict(self, X):
           return self.net_input(X)
       
      
X = df[['RM']].values
y = df['MEDV'].values
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

sns.reset_orig() # resets matplotlib style
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.savefig('/Users/mac/Desktop/Machine Learning/module4/Epoch.jpg')
plt.show()

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return None
lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.savefig('/Users/mac/Desktop/Machine Learning/module4/RM-MEDV-STD.jpg')
plt.show()

#RAMSAC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(LinearRegression(),
                          max_trials=100,
                          min_samples=50,
                          loss='absolute_loss',
                          residual_threshold=5.0,
                          random_state=0)
ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolor='white',
            marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolor='white',
            marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')
plt.savefig('/Users/mac/Desktop/Machine Learning/module4/RM-MEDV.jpg')
plt.show()

print('Slope: %.3f' % ransac.estimator_.coef_[0])
print('Intercept: %.3f' % ransac.estimator_.intercept_)
df = pd.read_csv('/Users/mac/Desktop/Machine Learning/module4/housing2.csv', header = 0)
df = df.dropna(axis=0)

#linear regression
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.3, random_state=0)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
#plot residuals versus predicted values
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.savefig('/Users/mac/Desktop/Machine Learning/module4/linear.jpg')
plt.show()

#compute the coefficient of determination
print('Slope:', slr.coef_)
print('Intercept: %.3f' % slr.intercept_)

#compute MSE for training and test predictions 
from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % (
       mean_squared_error(y_train, y_train_pred),
       mean_squared_error(y_test, y_test_pred)))

from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f' %
      (r2_score(y_train, y_train_pred),
       r2_score(y_test, y_test_pred)))
# the result indicates that the model is overfitting the training data

#Ridge regularization
from sklearn.linear_model import Ridge
#calculate the performance metrics
mse_train = []
mse_test = []
r2_train = []
r2_test =  []
for i in np.arange(0.0, 1.0, 0.1):
    ridge = Ridge(alpha=i)
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
    mse_train.append(mean_squared_error(y_train, y_train_pred))
    mse_test.append(mean_squared_error(y_test, y_test_pred))
    r2_train.append(r2_score(y_train, y_train_pred))
    r2_test.append(r2_score(y_test, y_test_pred))

for i in range(10):
    print('MSE train:', round(mse_train[i],6), 'test:', round(mse_test[i],6), 'R^2 train:', round(r2_train[i],6), 'test:', round(r2_test[i],6))

#the best performance in terms of alpha = 0
#plot residuals with alpha = 0.0
ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.savefig('/Users/mac/Desktop/Machine Learning/module4/Ridge_residulas.jpg')
plt.show()

#describe coefficients and y intercept
print('Slope:', ridge.coef_)
print('Intercept: %.3f' % ridge.intercept_)

#Lasso
from sklearn.linear_model import Lasso
mse_train = []
mse_test = []
r2_train = []
r2_test =  []
for i in np.arange(0.0, 1.0, 0.1):
    lasso = Lasso(alpha=i)
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    mse_train.append(mean_squared_error(y_train, y_train_pred))
    mse_test.append(mean_squared_error(y_test, y_test_pred))
    r2_train.append(r2_score(y_train, y_train_pred))
    r2_test.append(r2_score(y_test, y_test_pred))

for i in range(10):
    print('MSE train:', round(mse_train[i],6), 'test:', round(mse_test[i],6), 'R^2 train:', round(r2_train[i],6), 'test:', round(r2_test[i],6))

#the best performance in terms of alpha = 0.5
#plot residuals with alpha = 0.5
ridge = Lasso(alpha=0.5)
ridge.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.savefig('/Users/mac/Desktop/Machine Learning/module4/Lasso_residulas.jpg')
plt.show()

#describe coefficients and y intercept
print('Slope:', lasso.coef_)
print('Intercept: %.3f' % ridge.intercept_)

print("My name is {type your name here}")
print("My NetID is: {type your NetID here}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
