#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load libraries
import pandas as pd
from sklearn import preprocessing
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[2]:


Raw_data = pd.read_csv('Titanic_train.csv')
test = pd.read_csv('Titanic_test.csv')
Raw_data.head(10)

#Data cleaning (Missing Values)
Raw_data[Raw_data.isna().any(axis=1)]
data = Raw_data.drop(['Cabin'],axis=1)

data[data.isna().any(axis=1)]
data['Age'] = round(data['Age'].fillna(data['Age'].mean()))

data[data.isna().any(axis=1)]
data = data.drop([61,829])

data

#Ommitting unnecessary columns
data = data.drop(['PassengerId','Name','Ticket'],axis=1)
data.head()

#Sampling or Shuffling
data = data.sample(frac=1)

#Normalizing age
age = data[['Age']].values.astype(float)
age_scaled = preprocessing.MinMaxScaler().fit_transform(age)
data['normal_age'] = age_scaled

#Normalizing Fare
fare = data[['Fare']].values.astype(float)
fare_scaled = preprocessing.MinMaxScaler().fit_transform(fare)
data['normal_fare'] = fare_scaled

#Dropping original age and fare
data = data.drop(['Age','Fare'],axis=1)

#Categorical data encoding
codes = {'male':1, 'female':0}
data['Sex'] = data['Sex'].map(codes)

codes = {1:'I_class', 2:'II_class',3:'III_class'}
data['Pclass'] = data['Pclass'].map(codes)
pclass_dummies = pd.get_dummies(data.Pclass)
data = pd.concat([data,pclass_dummies],axis=1)
data = data.drop(['Pclass'],axis=1)


embarked_dummies = pd.get_dummies(data.Embarked)
data = pd.concat([data,embarked_dummies],axis=1)
data = data.drop(['Embarked'],axis=1)

#Splitting Data Set
x = data.iloc[:,1:13]
y= data.iloc[:,0].astype('category')
from sklearn.model_selection import train_test_split
X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size=0.2, random_state=1)


# In[3]:


# Different Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[ ]:




