# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 09:55:34 2019

Neural Network Regression using tensorflow. The NN will consist with one layer
and it will perform the weights and bias optimization. The is divided in 6
parts

The ex1_Data2.txt contains a training set of housing prices in Port-land,
Oregon. The first column is the size of the house (in square feet), the
second column is the number of bedrooms, and the third column is the price
of the house.

This code is based on the following works:
https://www.youtube.com/watch?v=WskWc15bcy4&index=8&list=LLu4xwOkmpJIwTC9SQj92e1g
https://github.com/antaloaalonso/Regression-Model-YT-Video

@author: gabi
"""
#%% Part 1 - importing libraries

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import pandas as pd

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from matplotlib import pyplot as plt
import MLplot as pl

#%% Part 2 - Importing and Organazing Data

# Path to data file
pathtodata = 'Exercise_Data/ex1_Data2.txt'

# Importing the Data as DataFrame (header = None)-not include the feature index
data = pd.read_csv(pathtodata,delimiter = ',',header=None)

x = data[[0,1]] # Size of the house and (X0) and number of bedrooms (X1)
y = data[[2]] # Price of houses in Port-land

#%% Part 3 - Split the dateset in training (80%) and testing (20%)

[X_train, X_test, y_train, y_test] = train_test_split(x,y,test_size=0.2,random_state=101)


#%% Part 4 -  Feature normalization in training and testing data

X_train = preprocessing.scale(X_train)

X_test = preprocessing.scale(X_test)

#%% Part 5 - Plotting the results for different learning rates

LR = [100,1000,10000] # Learning rate

for i in LR:
    
    #Defines linear regression model and its structure
    model = Sequential()
    model.add(Dense(1, input_shape=(2,)))
    
    #Compiles model
    model.compile(Adam(lr=i), 'mean_squared_error')
    
    #Fits model
    history = model.fit(X_train, y_train, epochs = 500, validation_split = 0.1,verbose = 0)
    history_dict=history.history
    
    #Plots model's training cost/loss and model's validation split cost/loss
    loss_values = history_dict['loss']
    val_loss_values  =history_dict['val_loss']
    plt.figure()
    plt.plot(loss_values,'bo',label='training loss')
    plt.plot(val_loss_values,'r',label='val training loss')

#%% Part 6 - Plot the performance with activation function   
    
model = Sequential()
model.add(Dense(1, input_shape=(2,), activation = 'relu'))
model.compile(Adam(lr=10000), 'mean_squared_error')
history = model.fit(X_train, y_train, epochs = 500, validation_split = 0.1,verbose = 0)

history_dict=history.history
loss_values = history_dict['loss']
val_loss_values=history_dict['val_loss']
plt.plot(loss_values,'bo',label='training loss')
plt.plot(val_loss_values,'r',label='training loss val')

#%% Part 7 - Run the NN in the test data set

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculates and prints r2 score of training and testing data
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))

#%% Part 8 - Run the NN in the test data set

# Defines "deep" model and its structure
model = Sequential()
model.add(Dense(13, input_shape=(2,), activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(1,))
model.compile(Adam(lr=0.003), 'mean_squared_error')

# Pass several parameters to 'EarlyStopping' function and assigns it to 'earlystopper'
earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

# Fits model over 2000 iterations with 'earlystopper' callback, and assigns it to history
history = model.fit(X_train, y_train, epochs = 2000, validation_split = 0.2,shuffle = True, verbose = 0, 
                    callbacks = [earlystopper])

# Plots 'history'
history_dict=history.history
loss_values = history_dict['loss']
val_loss_values=history_dict['val_loss']
plt.plot(loss_values,'bo',label='training loss')
plt.plot(val_loss_values,'r',label='training loss val')

# Runs model with its current weights on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculates and prints r2 score of training and testing data
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))

plt.plot(y_train, y_train_pred,'*r')
plt.plot(y_test, y_test_pred, '*g')
plt.figure()
for i in range(0,140):
    plt.plot(i/100,i/100,'*b')