#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 09:07:37 2023

Main script, Does not contain the neural network

@author: juanvargas
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import RocCurveDisplay ,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt 


os.getcwd()
training = pd.read_excel('GeneExpressionCancer_training.xlsx')
test_data = pd.read_excel('GeneExpressionCancer_test.xlsx')
val_data = pd.read_excel('GeneExpressionCancer_validation.xlsx')
##### Set up the data and pre process

#Training data
training_data = training.iloc[0:8000,0:500]
training_label = training['CancerDiagnosed']


X= training_data
y = training_label.astype('category') # Transform into categorical


#Test data

X_test = test_data.iloc[0:1000,0:500]
y_test = test_data['CancerDiagnosed']
y_test = y_test.astype('category')


pipeline = Pipeline([
    ('preprocessor', StandardScaler()),
])
pipeline.fit_transform(X) # Scale the data



######## Modeling Logistic

logistic_model = LogisticRegression(max_iter= 1000)
logistic_model.fit(pipeline.fit_transform(X),y) 

predicted_probas = logistic_model.predict_proba(pipeline.fit_transform(X_test))[:,1] #Positive class probabilities

fpr, tpr, thresholds = roc_curve(y_test, predicted_probas) #Using test dataset

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx] #Optimal is 0.538


RocCurveDisplay.from_predictions(
    y_test,
    predicted_probas,
    name="ROC",
    color="darkorange",
    plot_chance_level=True,
)


#On testing dataset


y_pred_new_threshold = (logistic_model.predict_proba(pipeline.fit_transform(X_test))[:, 1] >= 0.538).astype(int) 

accuracy_score(y_test, y_pred_new_threshold) # 0.546



        #probabilities of the positive label , test dataset

sns.histplot(predicted_probas, color='red', alpha=0.5, label='Positive Class') 

sns.histplot(logistic_model.predict_proba(pipeline.fit_transform(X_test))[:,0], alpha=0.5, label='Negative Class')


plt.legend(title='Class')
plt.show()



# On validatioon dataset
training_data = training.iloc[0:8000,0:500]
training_label = training['CancerDiagnosed']


X= training_data
y = training_label.astype('category') # Transform into categorical




X_val = val_data.iloc[0:1000,0:500]
y_val = val_data['CancerDiagnosed']
y_val = y_val.astype('category')


pipeline = Pipeline([
    ('preprocessor', StandardScaler()),
])
pipeline.fit_transform(X) # Scale the data



logistic_model = LogisticRegression(max_iter= 1000)
logistic_model.fit(pipeline.fit_transform(X),y) 

predicted_probas = logistic_model.predict_proba(pipeline.fit_transform(X_val))[:,1] #Positive class probabilities

fpr, tpr, thresholds = roc_curve(y_val, predicted_probas) #Using validation dataset

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx] 
print(optimal_threshold) # 0.6939509754712332

y_pred_new_threshold = (logistic_model.predict_proba(pipeline.fit_transform(X_val))[:, 1] >= 0.693).astype(int) 

acc = accuracy_score(y_val, y_pred_new_threshold)
print(acc) #0.522


RocCurveDisplay.from_predictions(
    y_val,
    predicted_probas, #Validation predicted probabilities
    name="ROC",
    color="darkorange",
    plot_chance_level=True,
)

plt.show()


sns.histplot(predicted_probas, color='red', alpha=0.5, label='Positive Class') 

sns.histplot(logistic_model.predict_proba(pipeline.fit_transform(X_val))[:,0], alpha=0.5, label='Negative Class')


plt.legend(title='Class')
plt.show()



####### Neural network





