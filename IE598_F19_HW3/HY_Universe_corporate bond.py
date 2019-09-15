#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 14:05:15 2019

@author: mac
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import dataset
data = open('/Users/mac/Desktop/Machine Learning/module3/HY_Universe_corporate bond.csv')

#Listing 2-1: Sising Up a New Data Set
#arrange data into list for labels and list of lists for attributes
xList = []
labels = []
for line in data:
    #split on comma
    row = line.strip().split(",")
    xList.append(row)
xList = xList[1:]
sys.stdout.write("Number of Rows of Data = " + str(len(xList)) + '\n')
sys.stdout.write("Number of Columns of Data = " + str(len(xList[1])) + '\n')
#Number of Rows of Data = 2721
#Number of Columns of Data = 37

#Listing 2-2: Determining the Nature of Attributes
nrow = len(xList)
ncol = len(xList[1])

type = [0]*3

colCounts = []

for col in range(ncol):
    for row in xList:
        try:
            a = float(row[col])
            if isinstance(a, float):
                type[0] += 1
        except ValueError:
            if len(row[col]) > 0:
                type[1] += 1
            else:
                type[2] += 1
    colCounts.append(type)
    type = [0]*3
sys.stdout.write("Col#" + '\t' + "Number" + '\t' + "Strings" + '\t ' + "Other\n")

iCol = 0
for types in colCounts:    
    sys.stdout.write(str(iCol) + '\t\t' + str(types[0]) + '\t\t' + str(types[1]) + '\t\t' + str(types[2]) + "\n")
    iCol += 1
    
#Listing 2-3:Summary Statistics for Numeric and Categorical Attributes
#the 10th column contains numerical variables, issued amount
col = 9
colData = []
for row in xList:
    colData.append(float(row[col]))
    
colArray = np.array(colData)
colMean = np.mean(colArray)
colsd = np.std(colArray)
sys.stdout.write("Mean = " + '\t' + str(colMean) + '\t\t' +
            "Standard Deviation = " + '\t ' + str(colsd) + "\n")
#calculate quantile boundaries
ntiles = 4
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
sys.stdout.write("\nBoundaries for 4 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")
#run again with 10 equal intervals
ntiles = 10
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
sys.stdout.write("Boundaries for 10 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")

#The 12th column contains categorical variables, maturity types
col = 11   
colData = []
for row in xList:
    colData.append(row[col])
unique = set(colData)
sys.stdout.write("Unique Label Values \n")
print(unique)
#count up the number of elements having each value
catDict = dict(zip(list(unique),range(len(unique))))
catCount = [0]*12
for elt in colData:
    catCount[catDict[elt]] += 1   
sys.stdout.write("\nCounts for Each Value of Categorical Label \n")
print(list(unique))
print(catCount)   

#Listing 2-4: Quantile‐Quantile Plot for attribute of issued amount
#regenerate colData for numerical attribute issued amount, overwritten by categorical statistic summary phase
import scipy.stats as stats
import pylab
col = 9
colData = []
for row in xList:
    colData.append(float(row[col]))
    
stats.probplot(colData, dist="norm", plot=pylab)
pylab.show()

#Listing 2-5: Using Python Pandas to Read and Summarize Data 
#read HY_Universe_corporate bond data into pandas data frame
bond = pd.read_csv('/Users/mac/Desktop/Machine Learning/module3/HY_Universe_corporate bond.csv',header=0)
#print head and tail of data frame
print(bond.head())
print(bond.tail())
#print summary of data frame
summary = bond.describe()
print(summary)

#Listing 2-6: Parallel Coordinates Graph for Real Attribute Visualization 
for i in range(2721):
    #assign color based on "AT MATURITY" and  "CALLABLE" or other labels
    if bond.iat[i,11] == "AT MATURITY":
        pcolor = "red"
    elif bond.iat[i,11] == "CALLABLE":
        pcolor = "blue"
    else:
        pcolor = 'yellow'
    #plot rows of data as if they were series data
    dataCol = bond.iloc[i,31:35]
    dataCol.plot(color = pcolor)
plt.xlabel("Attribute Index")
plt.ylabel(("Attribute Values"))
plt.show()    

#Listing 2-7: Cross Plotting Pairs of Attributes 
#calculate correlations between real-valued attributes "weekly_mean_volume", "weekly_median_volume", "n_trades"
dataCol32 = bond.iloc[0:270:,31]   
dataCol33 = bond.iloc[0:270:,32]
plt.scatter(dataCol32, dataCol33)
plt.xlabel("10th Attribute")
plt.ylabel("11th Attribute")
plt.show()
dataCol22= bond.iloc[0:270:,21]
plt.scatter(dataCol32, dataCol22)
plt.xlabel("10th Attribute")
plt.ylabel(("22nd Attribute"))
plt.show()
#It shows that "weekly_mean_volume" has strong correlation with "weekly_median_volume", 
#but nearly no correlation with "n_trades"  

#Listing 2-8: Correlation between Classification Target and Real Attributes     
#choose label in "Maturity Type", and attribute in "Coupon"
#change the targets to numeric values
from random import uniform
target = []
#Not use the whole rows considering the run time and the result, 
#becuase there are couples of point makes the plot scaler large
for i in range(272):
    #assign 0, 1, 2 target value based on "AT MATURITY", "CALLABLE" and other labels
    if bond.iat[i,11] == "AT MATURITY":
        target.append(1.0)
    elif bond.iat[i,11] == "CALLABLE":
        target.append(0.0)
    else:
        target.append(2.0)
#plot 9th attribute
dataCol = bond.iloc[:272,9]
plt.scatter(dataCol, target)
plt.xlabel("Attribute Value")
plt.ylabel("Target Value")
plt.show()

#To improve the visualization, this version dithers the points a little
#and makes them somewhat transparent
target = []
for i in range(272):
    #assign 0, 1, 2 target value based on "AT MATURITY", "CALLABLE" and other labels
    #and add some dither
    if bond.iat[i,11] == "AT MATURITY":
        target.append(1.0 + uniform(-0.1, 0.1))
    elif bond.iat[i,11] == "CALLABLE":
        target.append(0.0 + uniform(-0.1, 0.1))
    else:
        target.append(2.0 + uniform(-0.1, 0.1))
#plot 9th attribute with semi-opaque points
dataCol = bond.iloc[:272,9]
plt.scatter(dataCol, target, alpha=0.5, s=120)
plt.xlabel("Attribute Value")
plt.ylabel("Target Value")
plt.show()
#It shows that when the attribute is less than 4, the target would almost be CALLABLE, 
#otherwise, it can not be clarrified only by this attribute   

#Listing 2-9: Pearson’s Correlation Calculation for Attributes 31 versus 32 and 31 versus 10    
#calculate correlations between real-valued attributes
from math import sqrt
dataCol32 = bond.iloc[:,31]  
dataCol33 = bond.iloc[:,32]
dataCol11 = bond.iloc[:,10]
 
mean32 = 0.0; mean33 = 0.0; mean11 = 0.0
numElt = len(dataCol32)

for i in range(numElt):
    mean32 += dataCol32[i]/numElt
    mean33 += dataCol33[i]/numElt
    mean11 += dataCol11[i]/numElt
var32 = 0.0; var33 = 0.0; var11 = 0.0
for i in range(numElt):
    var32 += (dataCol32[i] - mean32) * (dataCol32[i] - mean32)/numElt
    var33 += (dataCol33[i] - mean33) * (dataCol33[i] - mean33)/numElt
    var11 += (dataCol11[i] - mean11) * (dataCol11[i] - mean11)/numElt
corr3233 = 0.0; corr3211 = 0.0
for i in range(numElt):
    corr3233 += (dataCol32[i] - mean32) * \
              (dataCol33[i] - mean33) / (sqrt(var32*var33) * numElt)
    corr3211 += (dataCol32[i] - mean32) * \
               (dataCol11[i] - mean11) / (sqrt(var32*var11) * numElt)
sys.stdout.write("Correlation between attribute 31 and 32 \n")
print(corr3233)
sys.stdout.write(" \n")
sys.stdout.write("Correlation between attribute 31 and 10 \n")
print(corr3211)
sys.stdout.write(" \n")
#Correlation between attribute 31 and 32 
#0.9567369663136258
 
#Correlation between attribute 31 and 10 
#0.3820504157175391
#It showa that attribute 31 and 32 have very strong correlation, and attribute 31 and 10
#may have certain correlation, but is much lower

#Listing 2-10: Presenting Attribute Correlations Visually and Listing 2-12: Correlation Calculation
#calculate correlations between real-valued attributes
from pandas import DataFrame
corMat = DataFrame(bond.corr())
#visualize correlations using heatmap
plt.pcolor(corMat)
plt.show()
print(corMat)

#Listing 2-11: Boxplot the  Data Set
#box plot the real-valued attributes
#choose 4 numerical attributes: "weekly_mean_volume", "weekly_median_volume"
#"weekly_max_volume", "weekly_min_volume"
#convert to array for plot routine
from pylab import boxplot
array = bond.iloc[:,31:35].values
boxplot(array)
plt.xlabel("Attribute Index")
plt.ylabel(("Quartile Ranges"))
plt.show()
#the last third atribute on the plot is out of scale with the rest remove and replot
array1 = bond.iloc[:,31:33].values
array2 = bond.iloc[:,34:35].values
array = np.hstack((array1,array2))
boxplot(array)
plt.xlabel("Attribute Index")
plt.ylabel(("Quartile Ranges"))
plt.show()
#This is another way to show the numerical attributes' statistical summaries 

print("My name is Pianpian Yu")
print("My NetID is: py7")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
