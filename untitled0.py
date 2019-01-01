#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:50:41 2018

@author: gabi
"""

import numpy as np
import pandas as pd

pathtodata = 'Exercise_Data/ex1_Data.txt'

# The pd.read_csv has many tools to manipulate the files. The delimiter for separate the collums is ',' and the example file does not contain collum index
# which pandas automaticaly put the first line as the index. For avoid this we use, header = None 

data = pd.read_csv(pathtodata,delimiter = ',',header=None) # data is DataFrame object (Table format)

# Variable with collum 1 index values OBS: Use two brackets to represent the 2D matrix correctly (Profit of the food truck of each city)
y = data[[1]]

x = []

for i in range(5):
    d = i*2
    x.append(d)

c = [-1.12541,2.246512]

z = np.dot(data,c)
    
# It is important to set the variables regarding your project. Following the course we can set (x,y) and the number of training examples
# Adding a new collum with only ones (For use Linear Regression later)

"""
    
    i = 0
for i in iterations:
    d = np.dot(x,t_cho)-y
    term_0 = d.mul(x.loc[:,"new"],axis=0)
    term_1 = d.mul(x.loc[:,0],axis=0)
    theta_0 = float(t_cho.loc[0,0] - ((alpha/m) * np.sum(term_0)))
    theta_1 = float(t_cho.loc[1,0] - ((alpha/m) * np.sum(term_1)))
    t_cho = pd.DataFrame([theta_0,theta_1])
    Jhist.loc[i,0] = float(cc.computeCost(x,y,t_cho))
    i = i + 1
    
    
"""