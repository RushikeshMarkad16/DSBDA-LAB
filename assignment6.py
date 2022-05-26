# -*- coding: utf-8 -*-
"""Assignment6.ipynb"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")

df.head()

df.variety.value_counts()

x=df.drop(['variety'],axis=1)
y=df['variety']

x.head()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state = 30)
print("Training, Testing and Splitting is successful")

from sklearn.naive_bayes import GaussianNB
 gnb = GaussianNB()
 gnb.fit(X_train, Y_train)

print("Training Accuracy:", gnb.score(X_train, Y_train)*100)

Y_predict = gnb.predict(X_test)
print("Testing Accuracy:", gnb.score(X_test, Y_test)*100)

from sklearn.metrics import accuracy_score
Acc = accuracy_score(Y_test, Y_predict)
print(Acc)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict)
print(cm)