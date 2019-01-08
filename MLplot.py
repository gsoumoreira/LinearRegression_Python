#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 17:20:34 2018

For plotting we will use matplotlib which is the most popular library for
cientific plot (It was designed to have similar feel to MatLab's graphical
plotting)

@author: gabi
"""

def plotstyle():
    
    import matplotlib.pyplot as plt
    
    # Creating a Figure obect (dpi changes the basic unit size)
    fig = plt.figure(figsize=(5,4),dpi=80) 

    # Settubg axis position and size
    axes = fig.add_axes([0,0,1,1])

    # Setting title and axis labels
    axes.set_title("2D plot")
    axes.set_xlabel("x values")
    axes.set_ylabel("y values")
    
    return [fig , axes]

def plot2D(x,y):
    
    import numpy as np
    import pandas as pd

    # Call the plotstyle function
    [fig,axes] = plotstyle()
    
    # Transforming in DataFrame
    x = pd.DataFrame(x)
    
    # Getting the Header information
    head = list(x.columns)
        
    # Calculate the number of features (collumns) and Training examples (rows)
    num_of_feat = len(x.columns)
        
    for i in range(num_of_feat):
        
        # Plotting the data
        axes.plot(x.iloc[:,i],y, label= head[i],
                  color=(np.random.sample(),np.random.sample(),
                  np.random.sample()),linewidth=0, linestyle='-',alpha=1,
                  marker='+', markersize=5, markeredgewidth=1)
                  
        # Inserting Legend
        axes.legend(loc=0)
    
    return axes

def plot3D(x,y,z):
    
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    from matplotlib import cm
              
    # Creating a Figure obect (dpi changes the basic unit size)
    fig = plt.figure(figsize=(8,7)) 
    axes = fig.gca(projection='3d')
    
    # Plotting the 3D surface graph data
    axes.plot_surface(x,y,z,linewidth=1, cmap=cm.coolwarm)
       
    # Setting title and axis labels    
    axes.set_xlabel('Parameter')
    axes.set_ylabel('Parameter')
    axes.set_zlabel('Parameter')
    
    
    ## Alternatives for the code
    
    fig.savefig('3Dplot.png') # - Save the figure
    # axes.set_xlabel(r'$\theta_0$') # - Using special characters
    # axes.view_init(60, 35) # - Chenge the view of the plot
    # ax.set_zlim(0, 800) # - Set the z range axis value
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f')) # - set the number style
    # fig.colorbar(surf, shrink=0.5, aspect=5) # Adding a colorbar legend
    
    
    return axes


def regressionPlot(x,y,theta,refcol):
    
    import numpy as np
    import pandas as pd

    # Call the plotstyle function
    [fig , axes] = plotstyle()
    
    # Transforming in DataFrame 
    x = pd.DataFrame(x)

    # Getting the Header information
    head = list(x.columns)
        
    # Setting the reference column for compute the hypothesis 
    x_ref = x.iloc[:,[0,refcol]]
            
    # Plotting the data
    axes.plot(x.iloc[:,refcol],y, label= head[refcol],
                  color=(np.random.sample(),np.random.sample(),
                  np.random.sample()),linewidth=0, linestyle='-',alpha=1,
                  marker='+', markersize=5, markeredgewidth=1)
    
    # Calculate the regression using the hypothesis theta
    reg = np.dot(x_ref,theta.iloc[[0,refcol],:])
    
    # Plotting the regression
    axes.plot(x.iloc[:,refcol],reg,label='Regression',color='blue',linewidth=1)

    # Setting title and axis labels
    axes.set_title('Regression plot')
    axes.set_xlabel('Parameter')
    axes.set_ylabel('Parameter')
   
    # Inserting Legend
    axes.legend(loc=0)
    
    # Saving Figure
    # fig.savefig('RegressionPlot.png')
    
    return axes

