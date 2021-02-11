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
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#encoding categorical data
from sklearn.preprocessing import  OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float) 

#avoiding dummy variable trap
X = X[:, 1:]

#splitting datatset into training aand the test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test =  train_test_split(X, y,test_size = 0.2, random_state= 0)


#fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set result
y_pred = regressor.predict(X_test)  
#building optimal model using backward elimination
import statsmodels.api as sm
X = np.append(arr=np.ones((len(X),1)).astype(int), values= X,axis=1)
"""X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y , exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y , exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog=y , exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog=y , exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog=y , exog=X_opt).fit()
regressor_OLS.summary()"""
def backwardElimination(x,sl):
    numVars = len(x[0])
    for i in range(0,numVars):
        regressor_OLS = sm.OLS(y,x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars-i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x=np.delete(x,j,1)
    regressor_OLS.summary()
    return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
