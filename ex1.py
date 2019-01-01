#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:26:13 2018

This file contains code that helps you get started on the
linear exercise. You will need to complete the following functions
in this exericse

x refers to the population size in 10,000s
y refers to the profit in $10,000s

@author: gabi
"""

#%% Part 1 - Importing the indenMatrix() function from warmUpExercise.py file

import warmUpExercise as wue

A = wue.idenMatrix() # A will receive the 2D 5x5 array identity

#%%  Part 2 - Importing and Organazing Exercise Data

# In python, is normally used pandas library to import csv, txt and some traditional data format.
# Pandas tranforms the data in a table with row and collums indexs
# It is important to install the recommended libraries (Using anaconda, for instance: conda install sqlalchemy, lxml, xlrd, BeautifulSoup4)

import numpy as np
import pandas as pd

pathtodata = 'Exercise_Data/ex1_Data.txt'

# The pd.read_csv has many tools to manipulate the files. The delimiter for separate the collums is ',' and the example file does not contain collum index
# which pandas automaticaly put the first line as the index. For avoid this we use, header = None 

data = pd.read_csv(pathtodata,delimiter = ',',header=None) # data is DataFrame object (Table format)

# Variable with collum 1 index values OBS: Use two brackets to represent the 2D matrix correctly (Profit of the food truck of each city)
y = data[[1]]
# Number of training examples 
m = len(y) 

# It is important to set the variables regarding your project. Following the course we can set (x,y) and the number of training examples
# Adding a new collum with only ones (For use Linear Regression later)
data["new"] = np.ones(m)

# Variable with collum index 0 values (Population of the Cities)
popci = pd.DataFrame(data.loc[:,0])

#%% Part 3 - Plotting the Exercising Data

# In the course, it was created a new function (plotData). However in Pyhton it is easy to import libraries. For plotting we will use matplotlib
# which is the most popular plotting library for python (It was designed to have similar feel to MatLab's graphical plotting)

import plotData as pl

pl.plot2D(popci,y)

#%% Part 3 - Cost and Gradient descent

#% Setting the Collum Ones as for the future Thetha 0 multiplication
x = data.loc[:,["new",0]]

# Important gradient descent settings
iterations = 1500
alpha = 0.01

# Initialize fitting parameters 
theta0 = pd.DataFrame([0,0])
theta1 = pd.DataFrame([-1,2])

# Compute and display initial cost (Choosing the theta values)

import computeCost as cc

J = cc.computeCost(x,y,theta1)

print("The cost function or total error using is = {0}".format(J))

# Run gradient descent

# Number of training examples (J_history)

import gradientDescent as gd

[bestHip,Jhist] = gd.gradientDescent(x,y,theta0,alpha,iterations)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5,4),dpi=80) 
axes = fig.add_axes([0,0,1,1])
axes.plot(popci,y, label='Data',color='red',linewidth=0,linestyle='--', alpha=1, marker='x', markersize=5, markerfacecolor='red', markeredgewidth=1, markeredgecolor='red')
axes.plot(popci,np.dot(x,bestHip), label='Data',color='blue',linewidth=1)


    
# Save the cost J in every iteration
#Jhist = cc.computeCost(x, y, theta);