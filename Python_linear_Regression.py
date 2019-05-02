# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 16:17:02 2019

@author: Yoga 910
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#FOR SIMPLE LINEAR REGRESSION WE DONT NEED SCALING. intead library will do it

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Use library to train the model

from sklearn.linear_model import LinearRegression
regressor=  LinearRegression()
regressor.fit(X_train, y_train)

#predection after training. We will perform Test model now

y_pred= regressor.predict(X_test)

# Plotting of predicted and actual results
plt.scatter(X_train,y_train, color='red')

#we are predicting actual salaries with  predicted salaries, so we use
#y=regressor.predict(X_train)
#red points are from training dat set
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title('Salary vs Experience(training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show() #to tell python that ploting code has ended

#now plot using test data to see the accuracy
plt.scatter(X_test,y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()













