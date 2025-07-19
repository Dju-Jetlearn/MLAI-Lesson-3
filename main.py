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
data["Embarked"] = label_encoder.fit_transform(data["Embarked"])

print(data.head())

x = data[["Pclass", "Sex", "Age", "Fare", "Embarked", "TravelAlone"]]
y = data["Survived"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

import seaborn as sans
from sklearn.metrics import confusion_matrix, accuracy_score

matrix = confusion_matrix(y_test, y_pred)

sans.heatmap(matrix, annot = True, fmt = "d")

matpat.title("Confusion Matrix")
matpat.xlabel("Predicted")
matpat.ylabel("Actual")
matpat.show()
acc = accuracy_score(y_test, y_pred)
print("accuracy = ", round((acc * 100), 2), "%")
