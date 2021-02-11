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