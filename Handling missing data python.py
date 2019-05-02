# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#set WD and import dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#seggregate X and Y variables

dataset= pd.read_csv('Data.csv')

#Always make sure to make x  a matrix and y  a vector
X= dataset.iloc[:, :-1].values
Y= dataset.iloc[:,3].values

#handling missing values by taking mean
#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer

imputer= Imputer(missing_values="NaN",strategy="mean",axis=0)
#fit this imputer to data X in which coloum it is required, not all coloumns
imputer=imputer.fit(X[:,1:3])
#replace missing data of X by coloumn mean
X[:, 1:3]= imputer.transform(X[:, 1:3])
##############################################################################

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
lbe_X= LabelEncoder()
X[:,0]= lbe_X.fit_transform(X[:,0])

#no need to run above

# So by endocing country names to 0,1,2 machine will think that 
# its a hirerchy in which 0 is less than 1, which is not true
#so to prevent this we will create a dummy variable, with three coloums
#of each country and each coloumn will have 1 or 0, 1 if country is present
#0 if country is not there

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
OHE=OneHotEncoder(categorical_features=[0])
X=OHE.fit_transform(X).toarray()

#now encode purchase coloumn. we do not need oneHotCode as its only Yes and No ategory
lbe_y= LabelEncoder()
Y= lbe_y.fit_transform(Y)


##############################################################################
#Spliting data for testing and trainning
# RandomState is for genertaing same answer. its like setting seed in R
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,
                                                 random_state=0) #0.2 is for test set

############################################################################
#Features scaling
#Standardisition=(X- mean(x))/sd(x)
#normalisation= (x-min(x))/max(x)- min(x)

from sklearn.preprocessing import StandardScaler
SC_X= StandardScaler()
X_train= SC_X.fit_transform(X_train)
#for test set dont need to fit it as it is already fitted to training set
#for better understanding of data you can only scale the last 2 coloums and not dummy variables
X_test= SC_X.transform(X_test)

#not for this case but when dependent variables has high set of values
#we need to scale dependent varible also









































