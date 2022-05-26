# -*- coding: utf-8 -*-
"""Assignment9.ipynb"""

import pandas as pd
import seaborn as sns

df=pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

df.head()

df.shape

df.Age.unique()

df.Survived.unique()

df.Sex.value_counts()

sns.countplot(data=df,x='Sex',hue='Survived')

sns.boxplot(data=df,x='Sex',y='Age',hue='Survived')

#We have  plotted the graph gender vs Age and setted hue as Survived. From above boxplot we can infer that

1. Maximum age in data is 40
2. Female count of survival is greater the=an male count of survival
3. Least age in the data is 18
4. Survived female age group is between 18 to 40
5. More older men died than younger men