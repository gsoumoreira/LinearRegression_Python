# -*- coding: utf-8 -*-
"""
COMPUTECOST Compute cost for linear regression
   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
   parameter for linear regression to fit the data points in X and y
"""

# x, y and theta should be a DataFrame Object 

def computeCost(x, y, theta):
    
    import numpy as np
    
    # Number of training examples
    m = len(y)
    
    # Initializing the cost variable 
    J = 0
    
    # Hypothesis function ((97x2) . theta(2x1) = theta_0 + theta_1 * x_1) 
    hip = np.dot(x,theta) 
    
    # Compute squared error for each y
    squaErr = np.power((hip - y),2)
    
    # The final cost Function
    J =  (1 / (2 * m)) * (np.sum(squaErr))
    
    return J