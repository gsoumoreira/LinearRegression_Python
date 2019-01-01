#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 18:31:37 2018

GRADIENTDESCENT Performs gradient descent to learn theta
theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
taking num_iters gradient steps with learning rate alpha

@author: gabi
"""

def gradientDescent(x, y, theta, alpha, num_iters):
    
    import numpy as np
    import pandas as pd
    import computeCost as cc
    
    # Initial parameters for gradient descent calculation
    i = 0
    m = len(y)
    iterations = list(range(num_iters))
    Jhist = pd.DataFrame(iterations)

    # Gradiante descent for iterations 
    for i in range(num_iters):

        # The general term for the gradient descent (hip*theta)-y
        gen_term = np.dot(x,theta)-y

        # Term for theta0 (gen_term*theta0)
        term_0 = gen_term.mul(x.loc[:,"new"],axis=0)

        # Term for theta1 (gen_term*theta1)
        term_1 = gen_term.mul(x.loc[:,0],axis=0)
        
        # Theta0 and theta 1 calculation
        theta_0= float(theta.loc[0,0] - ((alpha/m) * np.sum(term_0)))
        theta_1 = float(theta.loc[1,0] - ((alpha/m) * np.sum(term_1)))
       
        # Theta updating
        theta = pd.DataFrame([theta_0,theta_1])

        # Cost function History
        Jhist.loc[i,0] = float(cc.computeCost(x,y,theta))
        
        i = i + 1

    return [theta,Jhist]