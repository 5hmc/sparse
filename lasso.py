#/usr/bin/env python
from __future__ import print_function

## Lasso normalization program

import pandas as pd

#mat1 = pd.read_csv("exp_matrix_d.txt", header=None, sep=',')
mat1 = pd.read_pickle("mat1.pkl")
print('assign raw data to mat1 matrix')
print(mat1.shape)

print('-------------------------- Matrix count Finish !! ---------------------------------')

## slicing dataset matrix 

'''
mat2 = mat1.ix[:, mat1.columns != 'TCGA']
print('mat2 without TCGA column')
print(mat2.shape)
'''

label = mat1.iloc[:,-1]
'''
print('show only label column')
print(label)
'''

mat3 = mat1.iloc[:,1:-1]
print('show without 1st and last column')
print(mat3.shape)


print('-------------------------Prepared data set for lasso !! ---------------------------')

# Not run
from sklearn.cross_validation import train_test_split
from sklearn import linear_model

train_x, test_x, train_y, test_y  = train_test_split(mat3, label)

dic = zip(train_y.unique(), range(train_y.nunique()))
dic = dict(dic)

print(train_y)
train_y = map(lambda x: dic[x], train_y)
print(list(train_y))

regressor = linear_model.Lasso(alpha=1.0)
regressor.fit(train_x, list(train_y))

## show coefficient
print(regressor.coef_)


