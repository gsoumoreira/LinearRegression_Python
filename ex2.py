#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:26:13 2018

Exercise 2 -linear regression for n-features in pyhton. It will cover the
following parts:
    
   1 - Importing dataset
   2 - Feature normalization
   3 - Plotting the data normalized (data presentation)
   4 - Gradiente descent and compute cost function for n-features
   5 - Learning curve
   6 - Plotting the Linear Regression
   

The ex2_Data.txt contains a training set of housing prices in Port-land,
Oregon. The first column is the size of the house (in square feet), the
second column is the number of bedrooms, and the third column is the price
of the house.

@author: gabi
"""

#%%  Part 1 - Importing and Organazing Data

import pandas as pd 

# Path to data file
pathtodata = 'Exercise_Data/ex2_Data.txt'

# Importing the Data as DataFrame (header = None)-not include the feature index
data = pd.read_csv(pathtodata,delimiter = ',',header=None)

# Variable with collum 1 index values OBS: Use two brackets to represent the 2D matrix correctly (Profit of the food truck of each city)

# Size of the house and (X1) and number of bedrooms (X2)
x = data[[0,1]]
x.columns = ['X1','X2']

y = data[[2]] # Price of houses in Port-land

# Number of training examples 
m = len(y)

#%% Part 2 - Feature Scaling or Normalization

import featureNormalize as fn

[x_norm, x_mean, x_std] = fn.featureNormalize(x)

#%% Part 3 - Plotting the Data Normalized 

# Please check the MLplot source code for details
import MLplot as pl

dataPlot = pl.plot2D(x_norm,y)
dataPlot.set_title("Profit and Population")
dataPlot.set_xlabel('Size of the house and (X1) and number of bedrooms (X2)')
dataPlot.set_ylabel('Price of the house ($)')

#%% Part 4 - Gradient Descent for n-Features

import numpy as np
import gradientDescent as gd

# Inserting the first column X0 with the ones
x_norm.insert(loc=0, column='X0', value=np.ones(m))

# Calculate the number of features (columns) and rows (training examples)
num_of_feat = len(x_norm.columns)
num_of_train = len(x_norm)

# Choose the header feature index

# Variables for Gradient Descent
alpha = 0.1
num_iters = 400
theta_0 = pd.DataFrame(np.zeros([num_of_feat,1]))

# Gradient Descent and Cost Function History
[theta,Jhist,thetahist] = gd.gradientDescent(x_norm,y,theta_0,alpha,num_iters)

#%% Part 5 - Plotting the Learning Curve

# Iterations list values
iterations = pd.DataFrame(list(range(num_iters)))
iterations.columns = ['Iter']

# Plot the learning curve (alpha is the variable)
Leacur_plot = pl.plot2D(iterations,Jhist)
Leacur_plot.set_title(r'Learning curve for $\alpha$={0}'.format(alpha))
Leacur_plot.set_xlabel('Iterations')
Leacur_plot.set_ylabel('Cost Function')

#%% Part 6 - Plotting the Regression

# Check the MLplot source code for regressionPlot details
reg_plot = pl.regressionPlot(x_norm,y,theta,1)
reg_plot.set_title("Linear Regression")
reg_plot.set_xlabel('Size of the house normalized')
reg_plot.set_ylabel('Price of the house ($)')