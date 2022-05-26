
"""Data_Wrangling_Lab_Assignment_1.ipynb

"""
import pandas as pd

import matplotlib.pylab as plt

df=pd.read_csv("autodata.csv" )

# To see what the data set looks like, we'll use the head() method.
df.head(10)

df.tail(7)

df.info()

df.describe()

df.isnull()

df.isnull().sum()

df.notnull()

df.notnull().sum()

import numpy as np

# Write your code below and press Shift+Enter to execute

# calculate the mean vaule for "stroke" column
avg_stroke = df["stroke"].astype("float").mean(axis = 0)
print("Average of stroke:", avg_stroke)

# replace NaN by mean value in "stroke" column
df["stroke"].replace(np.nan, avg_stroke, inplace = True)

df['num-of-doors'].value_counts()

df['num-of-doors'].value_counts().idxmax()

#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace=True)

# simply drop whole row with NaN in "horsepower-binned" column
df.dropna(subset=["horsepower-binned"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)

df.isnull().sum()

df.dtypes

df.dtypes

# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

# check your transformed data 
df.head()

# Write your code below and press Shift+Enter to execute 
df['highway-L/100km'] = 235/df["highway-mpg"]

# check your transformed data 
df.head()

# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()

df['height'] = df['height']/df['height'].max() 
# show the scaled columns
df[["length","width","height"]].head()

df.columns

df['aspiration'].value_counts()

dummy_variable_1 = pd.get_dummies(df["aspiration"])
dummy_variable_1.head()

# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "aspiration" from "df"
df.drop("aspiration", axis = 1, inplace=True)

df.head()

df["horsepower"]=df["horsepower"].astype(float, copy=True)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins

group_names = ['Low', 'Medium', 'High']

df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)

df["horsepower-binned"].value_counts()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
