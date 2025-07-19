import numpy as np
import pandas as pd
import matplotlib.pyplot as matpat

data = pd.read_csv("C:/Users/diego/JetLearn/ML & AI/Lesson 3/winequality-red.csv")

print(data.isnull().sum())


print(data.head(20))

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
