import pandas as pd
import numpy as np
import matplotlib.cm as cm
import csv
import math 
import matplotlib.pyplot as plt
from sklearn import ensemble

print('Load data...')
train = pd.read_csv("../../../Input/train.csv")
target = train['target'].values
train = train.drop(['ID','target'],axis=1)

num_samples = train.shape[0];
num_variables = train.shape[1];
print('nsamples: ' , num_samples )

print('Counting nulls...')
var_nulls = np.zeros( num_variables )
var_types = np.zeros( num_variables )
xTickMarks = np.array(['var ' for _ in range(num_variables)])
counter = 0
for (train_name, train_series) in train.iteritems() :
	tmp_len = len(train[train_series.isnull()])
	var_nulls[counter] = tmp_len / num_samples 
	xTickMarks[counter] = train_name
	if train_series.dtype == 'O':
		var_types[counter] = 0
	else:
		var_types[counter] = 1
	if var_nulls[counter] < 0.2:
		print( 'var_name: ', train_name, ',  var_type:', train_series.dtype, ',    % of nulls: ', var_nulls[counter] )
	counter = counter + 1
	#print('train_name: ', train_name, '--- % of nulls: ', str( tmp_len / num_samples ) )

print('ticks: ' , xTickMarks)
fig = plt.figure()
ax = fig.add_subplot(111)
ind = np.arange(num_variables) 
width = 0.7
rects1 = ax.bar(ind, var_nulls, width,
                color='blue',
                error_kw=dict(elinewidth=2,ecolor='red'))

for x, y, bar in zip(ind, var_nulls, rects1):
	bar.set_facecolor(cm.jet(var_types[x]))
	bar.set_alpha(0.5)

ax.set_ylabel('% of nulls')
ax.set_title('Null values proportion')
ax.set_xticks(ind+width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, rotation=90, fontsize=8)
plt.axis([-1, 132, 0, 0.55])
print( 'close windows when finished... ')
plt.show()
