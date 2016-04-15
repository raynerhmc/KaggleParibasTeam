# Author: raynerhmc, modified version of kaggle user
# This implementation apply Gradient boost tree classifier over data with not too much processing
# All categorical data was converted to numerical through incremental numbers - 1 to 1 conversion.
# Catagorical data with null values were converted to -1
# Numerical data with null values were converted to -1001.

import pandas as pd
import numpy as np
import csv
import math 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn import metrics
import xgboost as xgb

print('Load data...')
train = pd.read_csv("../../../Input/jorge_train_001.csv")
target = train['target'].values
print ('target shape: ', target.shape)
train = train.drop(['ID','target'],axis=1)
test = pd.read_csv("../../../Input/jorge_test_001.csv")
id_test = test['ID'].values
test = test.drop(['ID'],axis=1)

nsamples = train.shape[0];
print('nsamples: ' , nsamples )

print('Clearing...')
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        #print('train_name: ', train_name)
        #print('train[train_name]: ', train[train_name])
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -1001
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -1001

cross_validation = 0.2
n_cv    = int(nsamples * cross_validation)
n_train = nsamples - n_cv

cv    = train[n_train:nsamples].copy()
new_train = train[:n_train].copy()
target_cv = target[n_train:nsamples].copy()
target_train = target[:n_train].copy()
print ('cv: ', cv.shape, '  --  cv-target: ', target_cv.shape)
print ('train: ', new_train.shape, '  --  train-target: ', target_train.shape)
xgtrain = xgb.DMatrix(new_train, target_train)
xgcv = xgb.DMatrix( cv, target_cv ) 
xgtest = xgb.DMatrix(test)


print('Training... ')

num_round = 500

params = {
        'objective':'binary:logistic',
        'eval_metric':'logloss',
        'eta' :             0.07,
        'min_child_weight': 1,
        'subsample':        0.9,
        'colsample_bytree': 0.9,
        'max_depth':        10,
    }

plst = list(params.items())
watchlist = [(xgtrain, 'train'),(xgcv, 'val')]
bst = xgb.train( plst, xgtrain, num_round, watchlist, early_stopping_rounds=50)

y_pred = bst.predict(xgtest)
print('Testing... ')

pd.DataFrame({"ID": id_test, "PredictedProb": y_pred}).to_csv('xgboost_test1.csv',index=False)


