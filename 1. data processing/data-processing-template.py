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