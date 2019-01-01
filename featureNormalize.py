#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 15:44:39 2018

function [X_norm, mu, sigma] = featureNormalize(X)
FEATURENORMALIZE Normalizes the features in X 
FEATURENORMALIZE(X) returns a normalized version of X where
the mean value of each feature is 0 and the standard deviation
is 1. This is often a good preprocessing step to do when
working with learning algorithms.

 Instructions: First, for each feature dimension, compute the mean
               of the feature and subtract it from the dataset,
               storing the mean value in mu. Next, compute the 
               standard deviation of each feature and divide
               each feature by it's standard deviation, storing
               the standard deviation in sigma. 

               Note that X is a matrix where each column is a 
               feature and each row is an example. You need 
               to perform the normalization separately for 
               each feature. 
               Instructions from:

@code author: gabi
"""

def featureNormalize(x):

    import pandas as pd
    import numpy as np

    # Calculate the number of features (columns)
    num_of_feat = len(x.columns)
    
    # Calculate the number of rows
    num_of_rows = len(x)
    
    # Initializing the variable as zeros DataFrame
    x_norm = pd.DataFrame(np.zeros([num_of_rows,num_of_feat]))
    
    # Mean and Standard Deviation of the the Features
    x_mean = pd.DataFrame(x.mean())
    x_std = pd.DataFrame(x.std())
   
    for i in range(num_of_feat):

        # The x input data normalized 
        x_norm.loc[:,i]  = (x.loc[:,i]-x_mean.loc[i,0])/(x_std.loc[i,0])

    return [x_norm, x_mean, x_std]

