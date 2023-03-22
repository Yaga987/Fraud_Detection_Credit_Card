import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('creditcard.csv')

"""
dataset from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

# Explore the data
print(data.head())
print(data.describe())

# Visualize the data
plt.hist(data['Class'])
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Undersample the majority class
fraud_data = data[data['Class'] == 1]
non_fraud_data = data[data['Class'] == 0].sample(n=len(fraud_data), random_state=42)
balanced_data = pd.concat([fraud_data, non_fraud_data])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(balanced_data.drop('Class', axis=1), balanced_data['Class'], test_size=0.2, random_state=42)

# Train a logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = lr.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))