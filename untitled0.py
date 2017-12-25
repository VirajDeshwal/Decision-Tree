#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:12:17 2017

@author: virajdeshwal
"""

import pandas as pd

file = pd.read_csv('Position_Salaries.csv')
X = file.iloc[:,1:2].values
y= file.iloc[:,2].values


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state = 0)


model.fit(X,y)

y_pred = model.predict(6.5)


#Visualizing the Decision tree in normal resolution

import matplotlib.pyplot as plt

plt.scatter(X,y, color = 'red')
plt.plot(X, model.predict(X), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#now as the decision tree is a non- continuous model. We have to use visualize the result in the higher resolution



import numpy as np

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y, color = 'cyan')
plt.plot(X_grid, model.predict(X_grid), color = 'green')
plt.xlabel('Position level')
plt.ylabel('Salary')

plt.show()
