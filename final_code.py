import pandas as pd
import numpy as np
import io
from sklearn.linear_model import Ridge

#importing the csv files
trainset = pd.read_csv('ad_org_train.csv')
testset = pd.read_csv('ad_org_test.csv')


#Converting the published  to date time
from datetime import datetime, timedelta
trainset['published'] = pd.to_datetime(trainset.published)
testset['published'] = pd.to_datetime(testset.published)
trainset['year'] = trainset.published.dt.year
trainset['month'] = trainset.published.dt.month
trainset['day'] = trainset.published.dt.day

testset['year'] = testset.published.dt.year
testset['month'] = testset.published.dt.month
testset['day'] = testset.published.dt.day

trainset['duration'] = trainset.duration.str.replace('PT','')
testset['duration'] = trainset.duration.str.replace('PT','')

trainset['duration'] = trainset.duration.str.replace('M','')
trainset['duration'] = trainset.duration.str.replace('H','')
trainset['duration'] = trainset.duration.str.replace('S','').astype(float)
testset['duration'] = testset.duration.str.replace('M','')
testset['duration'] = testset.duration.str.replace('H','')
testset['duration'] = testset.duration.str.replace('S','').astype(float)

#Processing the trainset for regression
X = trainset.drop(['vidid','adview','published',], axis=1)

#Applying getdummies for  converting categorical data in category column
X_train = X.join(pd.DataFrame(pd.get_dummies(X['category'])))
X_train= X_train.drop('category',axis = 1)

#Preprocessing the trainset data replacing invalid entries
X_train = X_train.interpolate()
X_train = X_train.replace('F',0).values
print(X_train)

#Processing the testset for regression
X_test = testset.drop(['vidid','published'],axis=1)

#Applying getdummies for converting categorical data in category column
X_test_pro = X_test.join(pd.DataFrame(pd.get_dummies(X_test['category'])))
X_test_pro= X_test_pro.drop('category',axis = 1)


#Preprocessing the testset data replacing invalid entries
X_test_pro = X_test_pro.interpolate()
X_test_pro = X_test_pro.replace('F',0).values 
Y_train = trainset['adview'].values

#Type casting the data to integer values from float
X_train = X_train.astype(int)
Y_train = Y_train.astype(int)
X_test_pro = X_test_pro.astype(int)

# Using ridge regularized regression technique
Ridgereg= Ridge(alpha = 0.6)
Ridgereg.fit(X_train,Y_train)
Y_test = Ridgereg.predict(X_test_pro)
Y_test = Y_test.astype(int)

#Calculating the ridge regression score
s = Ridgereg.score(X_test_pro,Y_test)
print(s)

#Type casting the adview to positive integers
Y_test = Y_test.astype(int)
Y_test[Y_test<0]=0

#A new  output dataframe in desired format
Vidid = testset['vidid'].values
df = pd.DataFrame({'vid_id':Vidid, 'ad_view':Y_test})
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
print(df.head())

#Exporting the dataframe to csv file
df.to_csv('output.csv')


