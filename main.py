# Logistic Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as matpat

data = pd.read_csv("C:/Users/diego/JetLearn/ML & AI/Lesson 3/titanic2.csv")

print(data.isnull().sum())

data["Age"].fillna(data["Age"].median(skipna = True), inplace = True)

data["Embarked"].fillna(data["Embarked"].value_counts().idxmax(), inplace = True)

print(data.isnull().sum())

data["TravelAlone"] = np.where((data["SibSp"] + data["Parch"]) > 0, 0, 1)

# Cabin, PassengerId, Name, Ticket, SibSp, Parch

data.drop('Cabin', axis = 1, inplace = True)
data.drop('PassengerId', axis = 1, inplace = True)
data.drop('Name', axis = 1, inplace = True)
data.drop('Ticket', axis = 1, inplace = True)
data.drop('SibSp', axis = 1, inplace = True)
data.drop('Parch', axis = 1, inplace = True)

print(data.head(20))

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

data["Sex"] = label_encoder.fit_transform(data["Sex"])
data["Embarked"] = label_encoder.fit_transform(data["Sex"])

print(data.head())