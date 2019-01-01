#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:26:13 2018

This file contains code that helps you get started on the
linear exercise in pyhton. It will cover the following parts:
    
    1 - Importing and arranging data
    2 - Feature normalization
    3 - Gradiente Descent for n-features (also computecost n-features)
    4 - Normal Equation

This file ex1data2.txt contains a training set of housing prices in Port-
land, Oregon. The first column is the size of the house (in square feet), the
second column is the number of bedrooms, and the third column is the price
of the house.

@author: gabi
"""

#%%  Part 1 - Importing and Organazing Exercise Data

'''
  In python, the pandas library can import different kind of file, such as: 
  csv, txt,etc. Even we can import in different ways, pandas is insteresting
  because its import as DataFrame. In DataFrame is easier to manipulate huge
  volume of data. Pandas tranforms the data in a table with row and collums
  indexs It is important to install the recommended libraries (Using anaconda, 
  for instance: conda install sqlalchemy, lxml, xlrd, BeautifulSoup4)
'''

import pandas as pd 

# Path to data file
pathtodata = 'Exercise_Data/ex1_Data2.txt'

# Importing the Data as DataFrame (header = None)-not include the feature index
data = pd.read_csv(pathtodata,delimiter = ',',header=None)

# Variable with collum 1 index values OBS: Use two brackets to represent the 2D matrix correctly (Profit of the food truck of each city)
x = data[[0,1]]
y = data[[2]]

# Number of training examples 
m = len(y)

#%% Part 2 - Feature Scaling or Normalization

import featureNormalize as fn

[x_norm, x_mean, x_std] = fn.featureNormalize(x)

#%% Part 3- Plotting the Data Normalized 
 
import plotData as pl

dataPlot = pl.plot2D(x_norm,y)
dataPlot.set_title("Profit and Population")

#%% Part 4 - Gradient Descent for n-Features

import numpy as np
import gradientDescent as gd

# Inserting the first column X0 with the ones
x_norm.insert(loc=0, column=2, value=np.ones(m))

# Calculate the number of features (columns) and rows (training examples)
num_of_feat = len(x_norm.columns)
num_of_train = len(x_norm)

# reseting the feature index
x_norm.columns = list(range(num_of_feat))

# Variables for Gradient Descent
alpha = 0.01
num_iters = 400
theta_0 = pd.DataFrame(np.zeros([num_of_feat,1]))

# Gradient Descent and Cost Function History
[theta,Jhist] = gd.gradientDescent(x_norm,y,theta_0,alpha,num_iters)

#%% Part 5 - Plotting the Learning Curve

# Iterations list values
iterations = pd.DataFrame(list(range(num_iters)))

# Plot the learning curve
plot = pl.plot2D(iterations,Jhist)

