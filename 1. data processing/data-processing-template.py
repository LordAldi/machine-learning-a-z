# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:12:09 2021

@author: User
"""

#data processing
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
Y = LabelEncoder().fit_transform(Y)

#splitting datatset into training aand the test set
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y,test_size = 0.2, random_state= 0)