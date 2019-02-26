# -*- coding: utf-8 -*-
"""
Created on Fri Feb  15 15:21:14 2019

@author: dparsara
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sklearn.metrics as metrics

# loading dataset
dataset = pd.read_csv("Credit Rating.csv", index_col = 0)

###############################################################################
# DATA EXPLORATION
###############################################################################
# check for null values
dataset.isnull().sum()

# check datatypes
dataset.dtypes

# check for outiers
sns.boxplot(x = "Gender", y = "Duration.of.Credit", data = dataset)
sns.boxplot(x = "Gender", y = "Amount.of.Credit", data = dataset)
sns.boxplot(x = "Credit.Rating", y = "Age", hue = "Gender", data = dataset)

# Payment of previous credits
payment_of_previous_credits_count = dataset["Payment.of.Previous.Credits"].value_counts()
sns.scatterplot(x = "Duration.of.Credit", y = "Amount.of.Credit", hue = "Occupation", data = dataset)

# Scaling: numeric columns
num_cols = dataset.columns[dataset.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
dataset[num_cols]
scaler = StandardScaler()
dataset[num_cols] = scaler.fit_transform(dataset[num_cols])

# Dummy variable creation
dummies_dataset = pd.get_dummies(dataset, drop_first = True)

# Target variable and Independent variables
target = dummies_dataset['Credit.Rating_good']
dependent = dummies_dataset.drop(['Credit.Rating_good'], axis = 1)

# Split dataset
xTrain, xTest, yTrain, yTest = train_test_split(dependent, target, test_size = 0.3, random_state = 1)

###############################################################################
# OPTIMAL K SELECTION BASED ON ACCURACY SCORES
###############################################################################
## We take a range of values for K(1 to 20) and find the accuracy 
## so that we can visualize how accuracy changes based on value of K
accuracy = []
for n in range(1,11):
    clf = KNeighborsClassifier(n_neighbors = n)
    clf.fit(xTrain,yTrain)
    y_pred = clf.predict(xTest)
    accuracy.append(accuracy_score(yTest,y_pred))
## Plotting the accuracies for different values of K
plt.figure(figsize=(16,9))
plt.plot(range(1,11),accuracy)

# Maximum accuracy 
accuracy.index(max(accuracy))

###############################################################################
# KNN MODEL BUILDING AND PREDICTIONS
###############################################################################
classifier = KNeighborsClassifier(n_neighbors = 8)
classifier.fit(xTrain, yTrain)

y_pred = classifier.predict(xTest)
y_pred = pd.DataFrame(y_pred)

###############################################################################
# MODEL EVALUATION
###############################################################################
# Confusion matrix
print(confusion_matrix(yTest, y_pred))

# ROC curve
# calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = metrics.roc_curve(yTest, y_pred)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Classification report
print(classification_report(yTest, y_pred, target_names = ['bad', 'good']))

# Accuracy score
accuracy_score(yTest, y_pred)

# Area under curve
metrics.roc_auc_score(yTest, y_pred)

###############################################################################
# CROSS VALIDATION
# 10 fold cross validation for selecting optimal K
###############################################################################
# creating odd list of K for KNN
myList = list(range(1,49))

# subsetting just the odd ones
neighbors = []
neighbors = filter(lambda x: x % 2 != 0, myList)
neighbors = list(neighbors)
neighbors

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, xTrain, yTrain, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    
# changing to misclassification error
MSE = [1 - x for x in cv_scores]
MSE = list(MSE)

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d", optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()