# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 03:11:50 2021

@author: User
"""

#polynomial regression
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#visualing dataset
"""plt.scatter(X, y, color='red')
plt.title("Salary vs level (dataset)")
plt.xlabel("level")
plt.ylabel("salary")
plt.show()"""

#fitting simple linear regressinon to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#fitting polynomial regressinon to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

#visualing the linear regression result
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X),color="yellow")
plt.title("Salary vs level (linear regression)")
plt.xlabel("level")
plt.ylabel("salary")
plt.show()
#visualing the polynomial regression result
X_grid = np.arange(min(X), max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)),color="blue")
plt.title("Salary vs level (polynomial regression)")
plt.xlabel("level")
plt.ylabel("salary")
plt.show()

#predicting a new result using linear regression
target = [[6.5]]
lin_reg.predict(target)

