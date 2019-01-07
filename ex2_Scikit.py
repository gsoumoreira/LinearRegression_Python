# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:39:41 2019

@author: gabi
"""

#%% Part 1 - importing scikit learn libraries

import MLplot as pl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import pandas as pd 

# Path to data file
pathtodata = 'Exercise_Data/ex1_Data2.txt'

# Importing the Data as DataFrame (header = None)-not include the feature index
data = pd.read_csv(pathtodata,delimiter = ',',header=None)

# Variable with collum 1 index values OBS: Use two brackets to represent the 2D matrix correctly (Profit of the food truck of each city)
x = data[[0,1]]
y = data[[2]]

min_max_scaler = preprocessing.MinMaxScaler()
x_norm = pd.DataFrame(preprocessing.scale(x))

pl.plot2D(x_norm,y)

# Number of training examples 
m = len(y)

[X_train, X_test, y_train, y_test] = train_test_split(x_norm,y,test_size=0.1,
random_state=101)

lm = LinearRegression()
lm.fit(X_train,y_train)
theta0 = pd.DataFrame(lm.intercept_)
theta = pd.DataFrame(lm.coef_)

x_norm.insert(loc=0, column="X0", value=np.ones(m))

theta.insert(loc=0, column="X0", value=theta0)

# Choose the header feature index
x_norm.columns = ['x0','x1','x2']

# Choose the header feature index
theta.columns = ['t0','t1','t2']


theta = theta.transpose()

predictions = lm.predict(X_test)

newpred = np.dot(x_norm,theta)

pl.regressionPlot(x_norm,y,theta,1)

