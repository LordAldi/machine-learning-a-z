# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 08:50:41 2021

@author: User
"""

#random forest regrsion 
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


#splitting datatset into training aand the test set
"""from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y,test_size = 0.2, random_state= 0)"""

#feature scalling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#fitting the random forest regrsion to the dataset
#create regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X,y)


#predicting a new result using random forest regrsion
y_pred = regressor.predict(np.array([[6.5]]))


#visualing the random forest regrsion result
X_grid = np.arange(min(X), max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid),color="blue")
plt.title("truth vs bluff random forest regrsion")
plt.xlabel("level")
plt.ylabel("salary")
plt.show()
