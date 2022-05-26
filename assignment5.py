# -*- coding: utf-8 -*-
"""Assignment5.ipynb"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/shivang98/Social-Network-ads-Boost/master/Social_Network_Ads.csv")

df.head

df.head(5)

df.shape

#Get input variable, features into X and outcome variable in Y

X = df.drop(['Gender', 'Purchased'], axis=1)
Y = df['Purchased']

X.head()

# Split the data into Train set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)
print("Training, Testing and Splitting is successful")

#Build Model
from sklearn.linear_model import LogisticRegression
basemodel = LogisticRegression()
basemodel.fit(X_train, Y_train)

print("Training Accuracy:", basemodel.score(X_train, Y_train)*100)

#Predict Data
Y_predict = basemodel.predict(X_test)
print("Testing Accuracy:", basemodel.score(X_test, Y_test)*100)

#Min Max Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = df[['Age', 'EstimatedSalary']]
X_scaled = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size = 0.2, random_state=42)
print("Normalization is Successful")

model = LogisticRegression()
model.fit(X_train, Y_train)
Y.predict = model.predict(X_test)
print("training Accuracy:", model.score(X_train, Y_train)*100)
print("Testing Accuracy:", model.score(X_test, Y_test)*100)

#Measure performance of model
from sklearn.metrics import accuracy_score
Acc = accuracy_score(Y_test, Y_predict)
print(Acc)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict)
print(cm)

from sklearn.metrics import classification_report
cr = classification_report(Y_test, Y_predict)
print(cr)