# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:53:08 2024

@author: PCC
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing the dataset
dataset = pd.read_csv('bank.csv')

# Taking care of missing data
# Replace "?" with np.nan in the specified columns
#columns_to_check = ["native.country", "workclass", "occupation"]
#dataset[columns_to_check] = dataset[columns_to_check].replace("?", np.nan)

# Drop rows with missing values in specified columns
#dataset_cleaned = dataset.dropna(subset=columns_to_check)

# Display the cleaned dataset
#print(dataset_cleaned)

# Extract X and y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder

# Specify the indices of the columns to be encoded
columns_to_encode = [1, 2, 3, 4,6, 7, 8, 10,15]

# Create a LabelEncoder for each column and apply the transformation to X
# Create a dictionary to store LabelEncoders for each column
label_encoders = {}

# Specify the indices of the columns to be encoded

# Create and apply the LabelEncoder for each column
for col in columns_to_encode:
    le = label_encoders.get(col, LabelEncoder())  # Retrieve existing encoder or create a new one
    unique_values = np.unique(X[:, col])  # Get unique values in the column
    unique_values.sort()  # Sort the unique values
    le.fit(unique_values)  # Fit the encoder with sorted unique values
    X[:, col] = le.transform(X[:, col])  # Apply the transformation
    label_encoders[col] = le  # Save the encoder to the dictionary

print(X)


# Encoding the Dependent Variable
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)





# Training the Decision Tree Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 90, criterion = "gini", random_state =42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
print(cm)
p=accuracy_score(y_test, y_pred)
print(p)
