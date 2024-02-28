# importing packages
import numpy as np
import pandas as pd
import numpy as np
from sklearn import preprocessing

# reading data
df = pd.read_csv('teleCust1000t.csv')
df.head()

# selecting features for x and y(base on our data)
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
y = df['custcat'].values

# normalizing data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# knn modeling
from sklearn.neighbors import KNeighborsClassifier
# starting algorithm by 4
k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
# predicting
yhat = neigh.predict(X_test)

# evaluation
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
