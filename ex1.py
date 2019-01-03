#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:26:13 2018

This file contains code that helps you get started on the
linear exercise in pyhton. It will cover the following parts:
    
   1 - WarmUpExercise - Importing files in python
   2 - Importing dataset
   3 - Plotting the data (data presentation)
   4 - Gradiente descent and compute cost function for one feature
   5 - Learning curv
   

The ex1_Data.txt contains information about population size in different 
citties (First column - numbers in 10.000s) and the profit of the company in
these citties (Second column - numbers in 10.000)

@author: gabi
"""

#%% Part 1 - Importing the indenMatrix() function from warmUpExercise.py file

import warmUpExercise as wue

A = wue.idenMatrix() # A will receive the 2D 5x5 array identity

#%%  Part 2 - Importing and Organazing Exercise Data

'''
  In python, the pandas library can import different kind of file, such as: 
  csv, txt,etc. Even we can import in different ways, pandas is insteresting
  because its import as DataFrame. Pandas tranforms the data in a table with 
  row and collums indexs It is important to install the recommended libraries
  (Using anaconda, for instance: conda install sqlalchemy, lxml, xlrd,
  BeautifulSoup4)
'''

import numpy as np
import pandas as pd


# Path for data file
pathtodata = 'Exercise_Data/ex1_Data.txt'

# Importing data using Pandas (Importing as DataFrame
data = pd.read_csv(pathtodata,delimiter = ',',header=None)

# Selecting the column variable with the profits 
y = data[[1]]

# Number of training examples 
m = len(y) 

# Adding a new column to the dataset with ones values
data["ones"] = np.ones(m)

# # Selecting the column variable with the population of citties 
popci = pd.DataFrame(data.loc[:,0])

#%% Part 3 - Plotting the Exercising Data

# MLplot source code for details
import MLplot as pl

pl.plot2D(popci,y)

#%% Part 4 - Cost funtion and gradient descent for one feature

#% Setting the Collum Ones as for the future Thetha 0 multiplication
x = data.loc[:,["ones",0]]

# Number of features (columns)
num_of_feat = len(x.columns)

# reseting the features index
x.columns = list(range(num_of_feat))

# Important gradient descent variables
iterations = 1500
alpha = 0.01

# Initialize fitting parameters 
theta0 = pd.DataFrame([0,0])

# Compute and display initial cost (Choosing the theta0 value)
import computeCost as cc

J = cc.computeCost(x,y,theta0)

# Gradient Descent and Cost Function History
import gradientDescent as gd

[bestHip,Jhist] = gd.gradientDescent(x,y,theta0,alpha,iterations)

#%% Part 4 - Plotting the linear regression

# Check the MLplot source code for details
import MLplot as pl

# Plotting the regression
regplot = pl.regressionPlot(x,y,bestHip,1)

