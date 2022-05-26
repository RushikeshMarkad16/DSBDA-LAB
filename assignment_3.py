# -*- coding: utf-8 -*-
"""Assignment_3.ipynb"""

# Import Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

df = pd.read_csv("Iris.csv")
df.head()

df.Species.value_counts()

# Let's check if we have something missing?
df.isnull().sum()

df.info()

feature = df.columns

plt.figure(figsize = (15, 5))
feature = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for i in enumerate(feature):
    plt.subplot(2,2,i[0]+1)
    sns.boxplot(df[i[1]])

sns.boxplot('sepal_width', data = df)

df1 = df.copy()

df.drop('id', axis = 1, inplace = True)

## Scaling
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
df2 = scale.fit_transform(df)

df2 = pd.DataFrame(df2)
df2.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df2.head()

## Hopkins Score

from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
 
def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H

hopkins(df2)

from sklearn.metrics import silhouette_score
ss = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters = k).fit(df2)
    ss.append([k, silhouette_score(df2, kmeans.labels_)])
    
plt.plot(pd.DataFrame(ss)[0], pd.DataFrame(ss)[1])

ssd = []
for k in range(2, 10):
    model= KMeans(n_clusters = k).fit(df2)
    ssd.append([k, model.inertia_])
    
plt.plot(pd.DataFrame(ssd)[0], pd.DataFrame(ssd)[1])

# Let's run kmean with 3

kmean = KMeans(n_clusters = 3, random_state = 100)
kmean.fit(df2)

df1.head()

df_km = pd.concat([df1, pd.Series(kmean.labels_)], axis =1)

df_km.head()

# Find the countries that are in need to aid based on 3 column, GDPP, Child_mort, Income

df_km.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'id', 'label']
df_km.head()

df_km.label.value_counts()

sns.scatterplot(x = "petal_length" , y = "petal_width", hue = 'label', data = df_km, palette = "Set1")

sns.scatterplot(x = "sepal_length" , y = "sepal_width", hue = 'label', data = df_km, palette = "Set1")

# GDPP, Child_mort, Income
# LOW GDPP
# High Child_mort
# Low Income
df_km[['petal_length', 'petal_width', 'label']].groupby("label").mean().plot(kind = 'bar')

df_km[df_km['label']==0].sort_values(by = ['petal_length', 'petal_width'], ascending = [True, True])

## Hirerachical clustering