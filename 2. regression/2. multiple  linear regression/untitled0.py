# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:19:24 2021

@author: User
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#splitting datatset into training aand the test set
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y,test_size = 0.2, random_state= 0)

#feature scalling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""