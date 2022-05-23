import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

train_df = pd.read_csv ('/content/train.csv')
display(train_df)
#NORMALIZING DATASET
train_df = pd.DataFrame(np.random.randint(1,100, 50).reshape(-1, 1))
train_norm = train_df.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))
train_norm 


train_df=pd.read_csv('/content/train.csv')

y = train_df.Cover_Type

X = train_df.drop('Cover_Type', axis=1)

#dividing Data Into 80%(Train) And 20%(test)
t_train, t_test, y_train, y_test = train_test_split(X, y,test_size=0.2)



#Loading Training Data From Drive
test=pd.read_csv('/content/test.csv')
test.head()

#---------------------DECISION TREE CLASSIFIER--------------------------------------------

clf = DecisionTreeClassifier(max_depth = 2).fit(t_train, y_train)
# accuracy on t_test
accuracy1 = clf.score(t_test, y_test)
print ("ACCURACY DECISION TREE CLASSIFERS:", accuracy1)
clf.fit(abs(t_train),y_train)
Cover_type=clf.predict(test)
print("Predicted Values: ",Cover_type)
# #Exporting The Two Colomns(Id And Cover_Type) Into exported colomn Csv
model_1_DTC = test[['Id']].copy()
model_1_DTC['Cover_Type'] = Cover_type
print(model_1_DTC)

"Creating Our Csv File With That Two Exported Columns For Submission On Kaggle"
model_1_DTC.to_csv('model_1_DTC.csv',index=False)
#----------------------NAIVE BAYES CLASSIFIER---------------------------------------------
# training a Naive Bayes classifier

clf = GaussianNB().fit(t_train, y_train)
# accuracy on t_test
accuracy2 = clf.score(t_test, y_test)
print ("ACCURACY OF NAIVE BAYES CLASSIFIERS:", accuracy2)
clf.fit(abs(t_train),y_train)
Cover_type = clf.predict(test)
print("Predicted Values: ",Cover_type)

# #Exporting The Two Colomns(Id And Cover_Type) Into exported colomn Csv
model_2_NB = test[['Id']].copy()
model_2_NB['Cover_Type'] = Cover_type
print(model_2_NB)

"Creating Our Csv File With That Two Exported Columns For Submission On Kaggle"
model_2_NB.to_csv('model_2_NB.csv',index=False)