import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
name = ['sepal length','sepal width','petal length','petal width','class']
dataset = pd.read_csv('iris.csv', names=name, header=None)

# split the data int x(training data) and y (results)
x = dataset.iloc[:,:-1].values
y = dataset['class'].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
array([[13,  0,  0],
       [ 0, 15,  1],
       [ 0,  1,  8]])
'''
#making the accuracy metric
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
''' ac = 0.9473684210526315 '''