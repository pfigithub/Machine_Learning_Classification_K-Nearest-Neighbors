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