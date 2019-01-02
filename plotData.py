#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 17:20:34 2018

For plotting we will use matplotlib which is the most popular library for
cientific plot (It was designed to have similar feel to MatLab's graphical
plotting)

@author: gabi
"""

def linRegplot(x,y,linReg=0):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    x = pd.DataFrame(x)
    linRegx = pd.DataFrame([0])
    
    # Calculate the number of features (collumns)
    num_of_feat = len(x.columns)
    
    # Creating a Figure obect (dpi changes the basic unit size)
    fig = plt.figure(figsize=(5,4),dpi=80) 

    # Settubg axis position and size
    axes = fig.add_axes([0,0,1,1])

    # Setting title and axis labels
    axes.set_title("Title 1")
    axes.set_xlabel("x values")
    axes.set_ylabel("y values")
    
    for i in range(num_of_feat):
        
        # Plotting the data
        axes.plot(x.loc[:,i],y, label='Data'+str(i),
                  color=(np.random.sample(),np.random.sample(),
                  np.random.sample()),linewidth=0, linestyle='-',alpha=1,
                  marker='+', markersize=5, markeredgewidth=1)
    
        # Inserting Legend (loc will set the legend position)
        axes.legend(loc=0)
        linRegx = pd.concat([linRegx,x.loc[:,i]])
        
    # Plotting the Linear Regression
    axes.plot(linRegx,linReg,label='Linear_Regression',color='blue',linewidth=1)
    axes.legend(loc=0)
    return axes

