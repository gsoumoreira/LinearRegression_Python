#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 17:20:34 2018

In Pyhton it is easy to import libraries. For plotting we will use matplotlib
which is the most popular plotting library for python (It was designed to have similar feel to MatLab's graphical plotting)

@author: gabi
"""

import matplotlib.pyplot as plt

def plotData(x,y):
    
# Creatting a Figure obect (Which is interesting for manipulation) dpi changes the basic unit size
    fig = plt.figure(figsize=(5,4),dpi=80) 

# This is will change the axis position and size! ([x point, y start point, x end point, y end point])
    axes = fig.add_axes([0,0,1,1])

# Setting title and axis labels
    axes.set_title("Trainning Examples ex1")
    axes.set_xlabel("x values")
    axes.set_ylabel("y values")
    
# Plotting the data - There are many options for graphs style, you can manipulate and change as you want
    axes.plot(x,y, label='Data',color='red',linewidth=0,linestyle='--', alpha=1, marker='x', markersize=5, markerfacecolor='red', markeredgewidth=1, markeredgecolor='red')
 
# Inserting Legend loc will set the position of the legend definning in plot(,label='Data')
    axes.legend(loc=0) 
    
    return

