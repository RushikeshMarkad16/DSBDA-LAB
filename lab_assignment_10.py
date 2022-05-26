# -*- coding: utf-8 -*-
"""Lab_Assignment_10.ipynb"""

# Import Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

df = pd.read_csv("Iris.csv")
df.head()

df.Species.value_counts()

df.drop('species', axis = 1, inplace = True)
df.head()

# Add a id column to this data, so that we can understand which folower belong to which cluster
df['id'] = df.index+100
df.head()

# Let's check if we have something missing?
df.isnull().sum()

df.info()

df.columns

for i in enumerate(feature):
    print(i)

plt.figure(figsize = (15, 15))
feature = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for i in enumerate(feature):
    plt.subplot(2,2,i[0]+1)
    sns.distplot(df[i[1]])

plt.figure(figsize = (15, 5))
feature = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for i in enumerate(feature):
    plt.subplot(2,2,i[0]+1)
    sns.boxplot(df[i[1]])

q1 = df['sepal_width'].quantile(0.01)
q4 = df['sepal_width'].quantile(0.99)

df['sepal_width'][df['sepal_width']<=q1]=q1
df['sepal_width'][df['sepal_width']>=q4]=q4

sns.boxplot('sepal_width', data = df)