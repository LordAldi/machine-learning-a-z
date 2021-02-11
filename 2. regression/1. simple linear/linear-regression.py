# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 12:59:24 2021
Simple Linear regression
@author: User
"""

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#splitting datatset into training aand the test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test =  train_test_split(X, y,test_size = 1/3, random_state= 0)

#fitting simple linear regressinon to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the Test set result
y_pred = regressor.predict(X_test)

#visualing the training set result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title("Salary vs Experiance (training set)")
plt.xlabel("years of experiance")
plt.ylabel("salary")
plt.show()
#visualing the test result  
plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title("Salary vs Experiance (test set)")
plt.xlabel("years of experiance")
plt.ylabel("salary")
plt.show()

