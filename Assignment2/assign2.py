# Importing libraries
import numpy as np #for matrices and vectors to work on linear algebra
import pandas as pd #use for data analysis


# Reading CSV File
trainDF = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')


# Extracting id from dataframe & pass into new dataframe object named as newDF
newDF = trainDF[['id']]
print(len(newDF))


# Add new target column to the position 1 of newDF object filled with zero's
newDF.insert(1,'target',0)
print(len(newDF))


# Generating random numbers against each row id in a target cell
newDF['target'] = np.random.rand(700000,1)


# Writing back to the csv file.
newDF.to_csv('out.csv', index=False)