#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 13:43:35 2018

GRADIENTDESCENT Performs gradient descent to learn theta
theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
taking num_iters gradient steps with learning rate alpha

@author: gabi
"""

def gradientDescent(x, y, theta, alpha, num_iters):
    
    import numpy as np
    import pandas as pd
    import computeCost as cc
    
    # Lenght of y
    m = len(y)
    
    # List witn the number of iterations
    iterations = list(range(num_iters))
    
    # Number of features
    num_of_feat = len(x.columns)
    
    # Gradient descent term initialization ((h(x)-y)*x)
    term = pd.DataFrame(np.zeros([m,num_of_feat]))
    
    # Jhist variable creation
    Jhist = pd.DataFrame(iterations)
    
    # thetahist variable creation
    thetahist = pd.DataFrame(np.zeros([num_iters,num_of_feat]))
    
    # Gradiante descent for iterations 
    for i in range(num_iters):

        # The general term for the gradient descent (hip*theta)-y
        gen_term = np.dot(x,theta)-y
        
        # Theta0 and theta 1 calculation
        for d in range(num_of_feat):
            
            # Element Wise multiplication (gen_term*theta1)
            term.loc[:,d] = gen_term.mul(x.iloc[:,d],axis=0)
        
            # Compute the gradient descent
            theta.iloc[d,:] =  theta.iloc[d,:] - ((alpha/m) * np.sum(term.iloc[:,d]))
        
            # Compute the theta history
            thetahist.iloc[i,d] = float(theta.iloc[d,:])
        
        # Cost function History
        Jhist.iloc[i,0] = float(cc.computeCost(x,y,theta))
        
    # Mean and Standard head index
    theta.columns = ["Theta"]
    Jhist.columns = ["Jhist"]
    
    return [theta,Jhist,thetahist]