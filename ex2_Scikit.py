# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:39:41 2019

Exercise 2 -linear regression for n-features in pyhton using scikit-learn. It
will cover the following parts:
    
   1 - Importing the necessary libraries and dataset
   2 - Feature normalization
   3 - Plotting the data normalized (data presentation)
   4 - Splitting the data into trainning and testing data
   5 - Setting the linear regression in scikit-learn
   6 - Plotting the Linear Regression

The ex2_Data.txt contains a training set of housing prices in Port-land,
Oregon. The first column is the size of the house (in square feet), the
second column is the number of bedrooms, and the third column is the price
of the house.

@author: gabi
"""

#%% Part 1 - importing libraries and dataset and organazing data

import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import MLplot as pl

# Path to data file
pathtodata = 'Exercise_Data/ex2_Data.txt'

# Importing the Data as DataFrame (header = None)-not include the feature index
data = pd.read_csv(pathtodata,delimiter = ',',header=None)

x = data[[0,1]] # Size of the house and (X0) and number of bedrooms (X1)
y = data[[2]] # Price of houses in Port-land

#%% Part 2 - Feature scalling using sklearn

# Calling the Feature scalling (preprocessing.scale)
x_norm = pd.DataFrame(preprocessing.scale(x))

#%% Part 3 - Plotting the data normalized

# Please check the MLplot source code for details
dataPlot = pl.plot2D(x_norm,y)
dataPlot.set_title("Profit and Population")
dataPlot.set_xlabel('Size of the house and (X1) and number of bedrooms (X2)')
dataPlot.set_ylabel('Price of the house ($)')

#%% Part 4 - Splitting data into trainning and testing data

# Number of data (Rows) 
m = len(y)

# Splitting the data
[X_train, X_test, y_train, y_test] = train_test_split(x_norm,y,test_size=0.3,
random_state=101)

#%% Part 5 - Setting the linear regression in scikit-learn

# Initialize linear regression
lm = LinearRegression()

# Fitting the trainning data
lm.fit(X_train,y_train)

# Compute the parameters (theta_0, theta_1 and theta_2)
theta0 = pd.DataFrame(lm.intercept_)
theta = pd.DataFrame(lm.coef_)

#%% Part 6 - Plotting the linear regression

# Preparing the hypothesis computation
x_norm.insert(loc=0, column="X0", value=np.ones(m))
theta.insert(loc=0, column="t0", value=theta0)

# Choose the header feature index
x_norm.columns = ['x0','x1','x2']

# Choose the header feature index
theta.columns = ['t0','t1','t2']

# Transpose theta (for pl.regressionPlot)
theta = theta.transpose()

# Plotting the linear Regression
reg_plot = pl.regressionPlot(x_norm,y,theta,1)
reg_plot.set_title("Linear Regression")
reg_plot.set_xlabel('Size of the house normalized')
reg_plot.set_ylabel('Price of the house ($)')