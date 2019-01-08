#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:26:13 2018

This file contains code for the ex1 - linear regression exercise in pyhton.
It will cover the following parts:
    
   1 - WarmUpExercise - Importing files in python
   2 - Importing dataset Organazing Data
   3 - Plotting the data (data presentation)
   4 - Gradiente descent and compute cost function for one feature
   5 - Estimate the profit for different values
   6 - Linear Regression plot
   7 - Cost Function plot with different theta values
   

The ex1_Data.txt contains information about population size in different 
citties (First column - numbers in 10.000s) and the profit of the company in
these citties (Second column - numbers in 10.000)

@author: gabi
"""

#%% Part 1 - Importing the indenMatrix() function from warmUpExercise.py file

import warmUpExercise as wue

A = wue.idenMatrix() # A will receive the 2D 5x5 array identity

#%%  Part 2 - Importing and Organazing Data

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

# Choose the header feature index
data.columns = ['X1','Y','X0']

# Selecting the column variable with the population of citties 
popci = pd.DataFrame(data.loc[:,'X1'])

#%% Part 3 - Plotting the Exercising Data

# MLplot source code for details
import MLplot as pl

pl.plot2D(popci,y)

#%% Part 4 - Cost funtion and gradient descent for one feature

# Setting the Features for linear regression
x = data.loc[:,['X0','X1']]

# Important gradient descent variables
iterations = 1500 # Number of iterations
alpha = 0.01 # Learning rate

# Initialize fitting parameters 
theta0 = pd.DataFrame([0,0])

# Compute and display initial cost (Choosing theta0)
import computeCost as cc

J = cc.computeCost(x,y,theta0)

# Gradient Descent, Cost Function History and Theta History
import gradientDescent as gd

[bestHip,Jhist,thetahist] = gd.gradientDescent(x,y,theta0,alpha,iterations)

#%% Part 5 - Estimate the profit for different x values

predict1 = np.dot(np.transpose(pd.DataFrame([1, 3.5])),bestHip)
predict2 = np.dot(np.transpose(pd.DataFrame([1,7])),bestHip)

#%% Part 6 - Plotting the Linear Regression

# Check the MLplot source code for regressionPlot details
regplot = pl.regressionPlot(x,y,bestHip,1)

# Setting the title and axis label
regplot.set_title('Linear Regression')
regplot.set_xlabel('Population in 10.000s')
regplot.set_ylabel('Profit in $10.000s')

#%% Part 7 - Cost Function plot with different theta values

# Setting theta_0 and theta_1 variation values
T0 = pd.DataFrame(np.linspace(-10,10,100))
T1 = pd.DataFrame(np.linspace(-1,4,100))

# Initialize the Cost function variable for 3D plot
J_vals = pd.DataFrame(np.zeros([len(T0),len(T1)]))

#Compute the new J_vals values
for i in list(range(len(T0))):
    for j in list(range(len(T1))):
         t = pd.DataFrame([float(T0.iloc[i,:]), float(T1.iloc[j,:])])
         J_vals.iloc[i,j] = float(cc.computeCost(x, y, t))

# Check the MLplot source code for plot3D details
Jplot = pl.plot3D(T0,T1,J_vals)

# Setting the title and axis label
Jplot.set_xlabel(r'$\theta_0$')
Jplot.set_ylabel(r'$\theta_1$')
Jplot.set_zlabel(r"J($\theta$)")