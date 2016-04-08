import pandas as pd
import numpy as np
import csv
import math 
from sklearn import ensemble
print('Load data...')
train = pd.read_csv("../input/without_nulls_train.csv")
target = train['target'].values
train = train.drop(['ID','target'],axis=1)

num_samples = train.shape[0];
print('nsamples: ' , num_samples )

print('Filling gaps and converting categorical variables to continuous...')
dic = {};
dic['class'] = target;
for (train_name, train_series) in train.iteritems():
    if train_series.dtype == 'O':
        #for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        dic[train_name] = train[train_name]
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -999 
        dic[train_name] = train[train_name]

print('saving to file...')
pd.DataFrame(dic).to_csv('train_filled_data.csv',index=False)