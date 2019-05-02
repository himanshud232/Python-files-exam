# Random Forest 
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn import metrics



# Importing the dataset


churn = pd.read_csv('Churn_Modelling.csv')
non_zero_mean = churn.Balance[churn.Balance != 0].mean()

churn.loc[churn.Balance == 0, "Balance"] = non_zero_mean


#balance mean
churn1=churn

non_zero_mean = churn1.Balance[churn1.Balance != 0].mean()

churn1.loc[churn1.Balance == 0, "Balance"] = non_zero_mean

#Class balancing

from sklearn.utils import resample

churn['Exited'].value_counts()

# Separate majority and minority classes
churn_majority = churn[churn.Exited==0]
churn_minority = churn[churn.Exited==1]
 
# Upsample minority class
churn_minority_upsampled = resample(churn_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=7963,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
churn_upsampled = pd.concat([churn_majority, churn_minority_upsampled])
 
# Display new class counts
churn_upsampled.Exited.value_counts()

X = churn_upsampled.iloc[:,:11].values
y = churn_upsampled.iloc[:,12].values

# Encoding Categorical Variable i.e Country

from sklearn.preprocessing import LabelEncoder
lbe_X= LabelEncoder()
X[:,3]= lbe_X.fit_transform(X[:,3])
print(X)

from sklearn.preprocessing import LabelEncoder
lbe_X= LabelEncoder()
X[:,4]= lbe_X.fit_transform(X[:,4])
print(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Random Forest classifier to the Training set
# Create your classifier here

from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators= 20,max_depth=15, criterion = 'entropy', random_state=0)
classifier.fit(X_train,y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



############ MC ###################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 
clf=RandomForestClassifier(n_estimators=20)
clf.fit(X_train,y_train)
rf_y_pred=clf.predict(X_test)
accu_rf=metrics.accuracy_score(y_test, rf_y_pred)
accu_rf

k=100
mc = np.zeros(7)
for j in range(1,10):
    result_array = np.array([])
    ratio=[0.1,0.2,0.4,0.5,0.6,0.7,0.8]
    for i in ratio:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=0)  
        clf=RandomForestClassifier(n_estimators=15)
        clf.fit(X_train,y_train)
        rf_y_pred=clf.predict(X_test)
        accu_rf=metrics.accuracy_score(y_test, rf_y_pred)
        result_array = np.append(result_array, accu_rf)
    mc=mc+(1/k)*result_array

# performance plot 

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.plot(ratio, mc, 'o', color='black');

#########################################
#n-Estimator
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
train_results = []
test_results = []
for estimator in n_estimators:
   rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D

line1= plt.plot(n_estimators,train_results,'b',label='Train AUC')
line2= plt.plot(n_estimators,test_results,'r',label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.show()
########################################
#Depth of trees

max_depths = np.linspace(1, 32, 50, endpoint=True)  
train_results1 = []
test_results1 = []
for max_depth in max_depths:
   rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results1.append(roc_auc)
   y_pred = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results1.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths,train_results1,'b',label='Train AUC')
line2, = plt.plot(max_depths,test_results1,'r',label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()

