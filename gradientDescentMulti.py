#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 13:43:35 2018

GRADIENTDESCENT Performs gradient descent to learn theta
theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
taking num_iters gradient steps with learning rate alpha

@author: gabi
"""

def gradientDescentMulti(x, y, theta, alpha, num_iters):
    
    import numpy as np
    import pandas as pd
    import computeCost as cc
    
    # Initial parameters for gradient descent calculation
    m = len(y)
    iterations = list(range(num_iters))
    Jhist = pd.DataFrame(iterations)
    num_of_feat = len(x.columns)
    term = pd.DataFrame(np.zeros([m,num_of_feat]))
    
    # Gradiante descent for iterations 
    for i in range(num_iters):

        # The general term for the gradient descent (hip*theta)-y
        gen_term = np.dot(x,theta)-y
        
        # Theta0 and theta 1 calculation
        for d in range(num_of_feat):
            
            # Wise multiplication (gen_term*theta1)
            term.loc[:,d] = gen_term.mul(x.loc[:,d],axis=0)
            theta.loc[d,:] =  theta.loc[d,:] - ((alpha/m) * np.sum(term.loc[:,d]))

        # Cost function History
        Jhist.loc[i,0] = float(cc.computeCost(x,y,theta))

    return [theta,Jhist]