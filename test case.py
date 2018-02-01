# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:14:42 2018

@author: c.luo
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(rc={'figure.figsize':(10, 10)}, font_scale = 1.25)
os.chdir('C:\Users\c.luo\Desktop\TestCase')
plt.style.use('seaborn-whitegrid')

###### Data preparation ######
### Import data
main = pd.read_csv('main.csv', sep = ';')
cust = pd.read_csv('cust.csv', sep = ';')
### Restruct cust-file
cust_new = cust.pivot(index = 'cust_id', columns = 'pref')
### Merge two files
main_merged = pd.merge(main, cust_new.reset_index(), on = 'cust_id')
### Regroup the target
main_merged['target'] = main_merged['boughtItem'].apply(lambda x: 1 if x == '1' else 0)
### Make a pivot table
main_merged.pivot_table(index = 'target')


###### Viz it! ######
plot_data = main_merged.drop(columns = ['boughtItem', 'cust_id', 'date_lastVisit', 'model1', 'model2', 'boughtItem_03', 'boughtItem_02', 'boughtItem_01'])

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for col in range(len(plot_data.columns)):
    if plot_data.columns[col] != 'target':
        fig.add_subplot(5, 6, col+1)
        sns.boxplot(x = 'target', y = plot_data.columns[col], data = plot_data)
    else:
        pass


###### Let's do the real thing ######
### Prepare for Scikit 
x_data = main_merged.drop(columns = ['target', 'boughtItem', 'cust_id', 'date_lastVisit', 'model1', 'model2', 'boughtItem_03', 'boughtItem_02', 'boughtItem_01'])
y_data = main_merged['target']
### Load some useful stuffs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
### Split the data for training and testing
xtrain, xtest, ytrain, ytest = train_test_split(x_data, y_data, test_size = 0.45, random_state = 0)
#### Naive Bayes - Gaussian ####
from sklearn.naive_bayes import GaussianNB
model_1 = GaussianNB()
model_1.fit(xtrain, ytrain)
model_1_pred = model_1.predict(xtest)
print(accuracy_score(ytest, model_1_pred))
#### Naive Bayes - multimonial ####
from sklearn.naive_bayes import MultinomialNB
model_11 = MultinomialNB()
model_11.fit(xtrain, ytrain)
model_11_pred = model_11.predict(xtest)
print(accuracy_score(ytest, model_11_pred))
#### Logistic Regression ####
from sklearn.linear_model import LogisticRegression
model_2 = LogisticRegression()
model_2.fit(xtrain, ytrain)
model_2_pred = model_2.predict(xtest)
print(accuracy_score(ytest, model_2_pred))

#### SVM ###
from sklearn.svm import SVC
model_3 = SVC(kernel = 'rbf', C = 1E6)
model_3.fit(xtrain, ytrain)
model_3_pred = model_3.predict(xtest)
print(accuracy_score(ytest, model_3_pred))


from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, model_3_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


