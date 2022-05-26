# -*- coding: utf-8 -*-
"""Assignment_8.ipynb"""

#importing pandas library
import pandas as pd
 
#loading data
titanic = pd.read_csv('train.csv')

# View first five rows of the dataset
titanic.head()

titanic.isnull().sum()

import seaborn as sns
import matplotlib.pyplot as plt
 
# Countplot
sns.catplot(x ="Sex", hue ="Survived",
kind ="count", data = titanic)

sns.barplot(x='Sex', y='Age', data=titanic)

sns.countplot(x = 'Pclass', hue = 'Survived', palette = 'Set2', data = titanic)

sns.countplot(x = 'Sex', hue = 'Survived', palette = 'Set1', data = titanic)

sns.countplot(x = 'Age',  data = titanic, hue = 'Survived', palette = 'Set1')

sns.countplot(x = 'Embarked', hue = 'Survived', palette = 'Set1', data = titanic)

sns.stripplot(x='Sex', y='Age', data=titanic , jitter=True )

sns.stripplot(x='Sex', y='Age', data=titanic , jitter=False)

sns.stripplot(x='Sex', y='Age', data=titanic, jitter=True, hue='Survived', split=True)

# Group the dataset by Pclass and Survived and then unstack them
group = titanic.groupby(['Pclass', 'Survived'])
pclass_survived = group.size().unstack()
 
# Heatmap - Color encoded 2D representation of data.
sns.heatmap(pclass_survived, annot = True, fmt ="d")

tc = titanic.corr()
sns.heatmap(tc,cmap='coolwarm')
plt.title('titanic.corr()')

# Violinplot Displays distribution of data
# across all levels of a category.
sns.violinplot(x ="Sex", y ="Age", hue ="Survived",
data = titanic, split = True)

# Adding a column Family_Size
titanic['Family_Size'] = 0
titanic['Family_Size'] = titanic['Parch']+titanic['SibSp']
 
# Adding a column Alone
titanic['Alone'] = 0
titanic.loc[titanic.Family_Size == 0, 'Alone'] = 1
 
# Factorplot for Family_Size
sns.factorplot(x ='Family_Size', y ='Survived', data = titanic)
 
# Factorplot for Alone
sns.factorplot(x ='Alone', y ='Survived', data = titanic)

sns.jointplot(x='Fare',y='Age',data=titanic)

sns.distplot(titanic["Fare"],kde=True)

sns.boxplot(x='Pclass',y='Age',data=titanic)

sns.swarmplot(x='Pclass',y='Age',data=titanic)

g = sns.FacetGrid(data=titanic, col='Sex')
g.map(sns.distplot, 'Age',kde=False)

sns.pairplot(titanic)

sns.pairplot(titanic, hue='Sex')

sns.histplot(titanic['Fare'], kde= True)