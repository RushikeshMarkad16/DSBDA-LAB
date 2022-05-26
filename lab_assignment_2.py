# -*- coding: utf-8 -*-
"""
# Lab Assignment 2
Data Wrangling II
Create an “Academic performance” dataset of students and perform the following operations
using Python."""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns # visualization library
# %matplotlib inline

df = pd.read_csv("/content/tecdiv.csv")

df.head()

df.describe()

df.info()

df.isnull().sum()

df.columns

sns.displot(df, x="First year:   Sem 1", kind="kde")

sns.displot(df, x="First year:   Sem 2")

sns.displot(df, x="Second year:   Sem 1" , kind = "kde")

sns.displot(df, x="Second year:   Sem 2")

sns.pairplot(df)

sns.boxplot(y=df['First year:   Sem 1'])

sns.boxplot(y=df['First year:   Sem 2'])

sns.boxplot(y=df["Second year:   Sem 1"])

sns.boxplot(y=df["Second year:   Sem 2"])

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['Roll no '], df['First year:   Sem 1'])
ax.set_xlabel('roll no of the students')
ax.set_ylabel('grades in sem 1')
plt.show()

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['Roll no '], df['First year:   Sem 2'])
ax.set_xlabel('roll no of the students')
ax.set_ylabel('grades in sem 2')
plt.show()

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['Roll no '], df['Second year:   Sem 1'])
ax.set_xlabel('roll no of the students')
ax.set_ylabel('grades in sem 3')
plt.show()

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['Roll no '], df['Second year:   Sem 2'])
ax.set_xlabel('roll no of the students')
ax.set_ylabel('grades in sem 4')
plt.show()