# importing required libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron


# Reading CSV files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#Extracting first 5 rows from train & head Dataframe
print(type(train))
print(type(test))
print("TRAIN")
print(train.head())
print("TEST")
print(test.head())

# Deleting id and f_27 column from train Dataframe & f_27 column from test Dataframe
del train['id']
del train['f_27']
del test['f_27']

#Data Preprocessing
X = train.drop(columns=['target'])
y = train['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Using Perceptron to train the following data
perceptron = Perceptron()
perceptron.fit(X_train, y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)
print(f"Accuracy = {acc_perceptron}")

per_clf = Perceptron()
per_scores = cross_val_score(per_clf, X_train, y_train, cv=6)
per_mean = per_scores.mean()
print('Naive Bayes Accuracy after CV: ',per_mean)

outputTest = test[['id']]
predT = test.drop(columns=['id'])
predictionOnTest = perceptron.predict(predT)
outputTest['target'] = predictionOnTest

# Saving Dataframe into new csv
outputTest.to_csv('outputTest.csv', index=False)
