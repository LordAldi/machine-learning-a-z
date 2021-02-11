# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 04:26:19 2021

@author: User
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


#feature scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1).astype(float))

#fitting the svr model to the dataset
#create regressor
from sklearn.svm import SVR
regressor = SVR(kernel = "rbf")
regressor.fit(X,y)



#predicting a new result using svr
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]])))) 



#visualing the svr result
X_grid = np.arange(min(X), max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid),color="blue")
plt.title("regression model")
plt.xlabel("level")
plt.ylabel("salary")
plt.show()