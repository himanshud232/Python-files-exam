# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 23:07:56 2019

@author: Yoga 910
"""

#NOTE: Always omit one dummy varible from model when making linear equation, 
#eg if you have 6 dummy variables, than add 5 only

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lbe_X= LabelEncoder()
X[:,3]= lbe_X.fit_transform(X[:,3])
OHE=OneHotEncoder(categorical_features=[3])
X=OHE.fit_transform(X).toarray()

#AVOIDING THE DUMMY VARIABLE TRAP : by removing one dummy variable from total dummy variablles
X=X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# NO NEED FOR FEATURE SCALING. LIBRARY WILL DO IT

#NOW FIT THE MODEL: Training set

from sklearn.linear_model import LinearRegression
Regressor= LinearRegression()
Regressor.fit(X_train, y_train)

#NOW FIT THE MODEL: Test set

y_pred= Regressor.predict(X_test)

#Using backward elimination
#statsmodel dosen't account for B0, so we habe to add it sperately
#we add a coloumn of values= 1 and attach it to X, to tell stats model about B0 constant
import statsmodels.formula.api as sm
X= np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)    #(50,1) becuase X has a size of 50 and 1 beacuse we need 1 colomn only

# backward elimination: create optimal matrix of features
#_opt contains only indept. variables which have high impact
#Type all columns as we have to remove them later
X_opt= X[:,[0,1,2,3,4,5]]

#BE:step1: select sig level
SL=0.05
#BE:Step 2 fit the model with all possible predictors
Regressor_ols= sm.OLS(endog=y,exog= X_opt).fit()
Regressor_ols.summary()
#The lower the P Value the higher the significant indep variable is.
#X2 has the highest P values , so we have to remove it
#Compare  X and X_opt  to remove the correct coloumn

X_opt= X[:,[0,1,3,4,5]]
Regressor_ols= sm.OLS(endog=y,exog= X_opt).fit()
Regressor_ols.summary()

#X1 has the highest P values , so we have to remove it
X_opt= X[:,[0,3,4,5]]
Regressor_ols= sm.OLS(endog=y,exog= X_opt).fit()
Regressor_ols.summary()

#X2 has the highest P values , so we have to remove it
X_opt= X[:,[0,3,5]]
Regressor_ols= sm.OLS(endog=y,exog= X_opt).fit()
Regressor_ols.summary()

#X2 has the highest P values , so we have to remove it
X_opt= X[:,[0,3]]
Regressor_ols= sm.OLS(endog=y,exog= X_opt).fit()
Regressor_ols.summary()

#Above is the Finalize version











