# -*- coding: utf-8 -*-
"""Asignment7.ipynb"""

str="My name is Khan, I am not terrorist"

import nltk
nltk.download('all')
from nltk.tokenize import word_tokenize
token=word_tokenize(str)
print(token)

from nltk.stem import PorterStemmer
ps = PorterStemmer()
 
# choose some words to be stemmed
words = ["Chocolate", "drawer", "huts", "programming", "programmers"]
 
for w in words:
    print(w, " : ", ps.stem(w))

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
for w in token:
  print(lemmatizer.lemmatize(w))

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered_sentence = [w for w in token if not w.lower() in stop_words]
 
filtered_sentence = []
for w in token:
    if w not in stop_words:
        filtered_sentence.append(w)
print(filtered_sentence)