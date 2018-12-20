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

# It is important to set the variables regarding your project. Following the course we can set (x,y) and the number of training examples

x = data[0] # Variable with collum 0 values
y = data[1] # Variable with collum 1 values
m = len(y) # Number of training examples


#%% Part 3 - Plotting the Exercising Data

# In the course, it was created a new function (plotData). However in Pyhton it is easy to import libraries. For plotting we will use matplotlib
# which is the most popular plotting library for python (It was designed to have similar feel to MatLab's graphical plotting)

import plotData as pl

pl.plotData(x,y)

#%% Part 3 - Cost and Gradient descent

#% Add a column of ones to x
newcol = np.ones(m)
newX = x.insert(newcol)

# Initialize fitting parameters




