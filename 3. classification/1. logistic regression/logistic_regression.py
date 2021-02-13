# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 11:37:25 2021

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


#splitting datatset into training aand the test set
"""from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y,test_size = 0.2, random_state= 0)"""

#feature scalling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#fitting the regressinon model to the dataset
#create regressor



#predicting a new result using polynomial regression
y_pred = regressor.predict(np.array([[6.5]]))


#visualing the regression result
X_grid = np.arange(min(X), max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid),color="blue")
plt.title("regression model")
plt.xlabel("level")
plt.ylabel("salary")
plt.show()
