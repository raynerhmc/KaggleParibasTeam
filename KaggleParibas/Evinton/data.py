import pandas as pd
import operator
import numpy as np
from scipy import stats
import csv
import math
from sklearn import svm
from sklearn.decomposition import PCA, KernelPCA
from operator import itemgetter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn import metrics
import xgboost as xgb

class KaggleProcess(object):
	"""KaggleProcess"""
	__var_cat = ['v3','v22','v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107' , 'v110', 'v112', 'v113', 'v125']
	__var_num = ["v1","v2","v4","v5","v6","v7","v8","v9",
				   "v10","v11","v12","v13"   
				  "v14","v15","v16","v17"   
				  "v18","v19","v20","v21"   
				  "v23","v25","v26","v27"   
				  "v28","v29","v32","v33"   
				  "v34","v35","v36","v37"   
				  "v38","v39","v40","v41"   
				  "v42","v43","v44","v45"   
				  "v46","v48","v49","v50"   
				  "v51","v53","v54","v55"   
				  "v57","v58","v59","v60"   
				  "v61","v62","v63","v64"   
				  "v65","v67","v68","v69"   
				  "v70","v72","v73","v76"   
				  "v77","v78","v80","v81"   
				  "v82","v83","v84","v85"   
				  "v86","v87","v88","v89"   
				  "v90","v92","v93","v94"   
				  "v95","v96","v97","v98"   
				  "v99","v100","v101","v102"  
				  "v103","v104","v105","v106"  
				  "v108","v109","v111","v114"  
				  "v115","v116","v117","v118"  
				  "v119","v120","v121","v122"  
				  "v123","v124","v126","v127"  
				  "v128","v129","v130","v131"] 

	def __init__(self):
		print('Read Data')
		self.__train = pd.read_csv('../input/train.csv')
		self.__test = pd.read_csv('../input/test.csv')
		#save the ids and target
		self.__ids = self.__train['ID'].values
		self.__targets = self.__train['target'].values
		self.__idstest = self.__test['ID'].values
		print('drop corrleted variable')
		#drop it 
		self.__train = self.__train.drop( ['ID','target','v46', 'v63', 'v17', 'v60', 'v48', 'v100', 'v115'], axis = 1)
		self.__test = self.__test.drop(['ID','v46', 'v63', 'v17', 'v60', 'v48', 'v100', 'v115'], axis = 1)

		self.n_samples = self.__train.shape[0]
		self.n_features = self.__train.shape[1]

		self.__data = self.__train
		self.__datatest = self.__test

		self.__fill = False
		print('Processing ',self.n_samples * self.n_features," data")

	def __fill_data_mean(self):
		'''
		filling the data with means value for the numerical variable and mode value for the categorical variable
		'''
		dic = {}
		dictest = {}
		#handle the moda value for the categorical name
		for (feature_name, feature_samples),(test_name, test_series) in zip(self.__data.iteritems(),self.__datatest.iteritems()):
			if feature_samples.dtype == 'object':
				#moda
				
				self.__data.loc[feature_samples.isnull(),feature_name] = feature_samples.mode()[0]
				self.__datatest.loc[test_series.isnull(),test_name] = feature_samples.mode()[0]
				#factor
				#self.__data.loc[feature_name],  tmp_indexer= pd.factorize(self.__data[feature_name]) 
				#self.__datatest[test_name] = tmp_indexer.get_indexer(self.__datatest[test_name])
				#self.__data.loc[feature_samples.isnull(),feature_name] = feature_samples.mode()[0]
				#self.__train.loc[feature_samples.isnull(),feature_name]= stats.mode(feature_samples)
			else:

				tmp_len = len(self.__data[feature_samples.isnull()])
				if tmp_len>0:
					self.__data.loc[feature_samples.isnull(), feature_name] = np.mean(feature_samples)
					self.__datatest.loc[test_series.isnull(), test_name] = np.mean(feature_samples) 
			
			dic[feature_name] = self.__data[feature_name]
			dic[feature_name] = self.__data[feature_name]
		self.__facotrize_categorical()

		#self.__data = self.__DataFrame(self.__data)
		#self.__datatest = self.__DataFrame(self.__datatest)
		return self.__data

	def __facotrize_categorical(self):
		for (feature_name, feature_samples),(test_name, test_series) in zip(self.__data.iteritems(),self.__datatest.iteritems()):
			if feature_samples.dtype == 'object':
				self.__data[feature_name], tmp_indexer = pd.factorize(self.__data[feature_name])
				self.__datatest[test_name] = tmp_indexer.get_indexer(self.__datatest[test_name])

	def __fill_null_level(self):
		'''
		filling the data with means value for the numerical variable and new level 'null' for the categorical variable
		'''
		dic = {}
		#handle the moda value for the categorical name
		for (feature_name, feature_samples),(test_name, test_series) in zip(self.__data.iteritems(),self.__datatest.iteritems()):
			if feature_samples.dtype == 'object':
				self.__data.loc[feature_samples.isnull(),feature_name] = 'null'
				self.__datatest.loc[test_series.isnull(),test_name] = 'null'
				#self.__train.loc[feature_samples.isnull(),feature_name]= stats.mode(feature_samples)
		
	def __fill_data_dummy(self):
		dic = {}
		for (feature_name, feature_samples) in self.__train:
			print(feature_name)

	def __fill_data(self):
		'''
		filling data code Rayner, returns one DataFrame
		'''
		dic = {};
		#dic['class'] = self.__targets;
		for (train_name, train_series), (test_name, test_series) in zip(self.__data.iteritems(),self.__datatest.iteritems()):
			
			if train_series.dtype == 'object':

				#for objects: factorize
				self.__data[train_name], tmp_indexer = pd.factorize(self.__data[train_name])
				self.__datatest[test_name] = tmp_indexer.get_indexer(self.__datatest[test_name])
				
				dic[train_name] = self.__data[train_name]
        		#but now we have -1 values (NaN)
			else:
				#for int or float: fill NaN
				tmp_len = len(self.__data[train_series.isnull()])
				
				if tmp_len>0:
					#print "mean", train_series.mean()
					self.__data.loc[train_series.isnull(), train_name] = -999

				tmp_len = len(self.__datatest[test_series.isnull()])

				if tmp_len>0:
					self.__datatest.loc[test_series.isnull(), test_name] = -999
				
				dic[train_name] = self.__data[train_name]

		return self.__data

	def fill_data(self,tfill = 'data'):
		self.__fill = True
		print("Filling the data")
		self.__fill_null_level()
		opt = {
			'dummy':self.__fill_data_dummy,
			'mean': self.__fill_data_mean,
			'data':	self.__fill_data
		}
		if(tfill in opt ): 
			#self.__data = 
			opt[tfill]()

	def drop_correlated(self):
		"""
		returns correlated variable Jarlinton Code
		"""
		corrleted = {}
		if not self.__fill:
			self.fill_data()
		# Compute the correlation matrix
		corr = self.__data.corr()
		threshold = 0.9 # correlation positive upper .9 degree
		limit = 1 # this is the same variable 
		
		for i in range( 0,corr.shape[0] ):
			lista = [];
			for j in range( 0, corr.shape[1] ):
			# correlation over 90% and under 100%
				feat =  corr.columns.values[j]
				if corr.iloc[i][j] > threshold and i != j:
					if(feat not in corrleted):
						corrleted[feat] = 1
					else:
						corrleted[feat] = corrleted[feat]  + 1
					lista.append(feat)
				elif corr.iloc[i][j] < -threshold :
					lista.append(feat)
					if(feat not in corrleted):
						corrleted[feat] = 1
					else:
						corrleted[feat] = corrleted[feat]  + 1

			if len(lista) > 0 :
				feat =  corr.columns.values[i]
				
		
		#corrleted = sorted(corrleted.items(),key=itemgetter(1), reverse=True)
		self.__dropvariable(['v46', 'v63', 'v17', 'v60', 'v48', 'v100', 'v115'])
		#return corrleted.keys()
					# print only one correlation example 2x3 and not 3x2 because it's the same.
					#print ('positive correlation between: v', str(i), ' and v' , str(j) , ' --> corr: ', corr.iloc[i][j] )
					 	
	def __numcorrelated(self):
		return True

	def __catcorrelated(self):
		return True

	def __dropvariable(self,data):
		self.__data = self.__data.drop(data,axis=1)


	def fecture_selection(self,feat):
		opt = {
			'l1':self.l1,
			'tree' : self.tree
		}
		if(feat in opt ): 
			#self.__data = 
			opt[feat]()

	def  l1(self):
		'''
		data =  self.__DataFrame(data)
		mn = data.shape[0] - self.n_samples
		mn = data.shape[0] - mn 
		self.__data = data[:self.n_samples].copy()
		l = self.__idstest.shape[0]
		self.__datatest = data[-l:data.shape[0]].copy()
'''
		frame = [self.__data,self.__datatest]
		result = pd.concat(frame)
		lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(self.__data, self.__targets)
		
		model = SelectFromModel(lsvc, prefit=True)
		X_new = model.transform(self.__data)
		print(lsvc.coef_)
		self.__data = self.__DataFrame(X_new)
		#print(self.__data.columns.values)

	def tree(self):
		clf = ExtraTreesClassifier()
		clf = clf.fit(self.__data, self.__targets)
		print(clf.feature_importances_)

	def get_pca(self,ncomponete=None):
		"""
		this function returns the array variables, for to use in the classifier
		"""
		# fit the model to data 
		#pca = PCA(n_components=ncomponete)
		if (not ncomponete):
			ncomponete = min(self.n_samples,self.n_features)

		
		pca = PCA(n_components=ncomponete)
		frame = [self.__data,self.__datatest]
		result = pd.concat(frame)
		data = pca.fit_transform(result)		
		data =  self.__DataFrame(data)
		mn = data.shape[0] - self.n_samples
		mn = data.shape[0] - mn 
		self.__data = data[:self.n_samples].copy()
		l = self.__idstest.shape[0]
		self.__datatest = data[-l:data.shape[0]].copy()
		
		#self.__data = pca.fit_transform(self.__data)
		#self.__datatest = pca.fit_transform(self.__datatest)
		#self.__datatest = pca.fit_transform(self.__datatest)

		#self.__data = self.__DataFrame(self.__data)
		#self.__datatest = self.__DataFrame(self.__datatest)
		
		dicdata = {}
		dictest = {}
		dicdata['ID'] = self.__ids
		dicdata['target'] = self.__targets
		
		dictest['ID'] =  self.__idstest
		
		for (feature_name, feature_samples) in self.__datatest.iteritems():
			dictest[feature_name] = feature_samples

		for (feature_name, feature_samples) in self.__data.iteritems():
			dicdata[feature_name] = feature_samples

	
		pd.DataFrame(dicdata).to_csv('../Input/without_nulls_train.csv',index=False)

		pd.DataFrame(dictest).to_csv('../Input/without_nulls_test.csv',index=False)
		#transform the data '''
	def get_kpca(self):
		"""
		this function returns the array variables, for to use in the classifier
		"""
		kpca = KernelPCA(kernel="poly", fit_inverse_transform=True, gamma=1)
		self.__data = kpca.fit_transform(self.__data)
		self.__data =self.__DataFrame(self.__data)

		dicdata = {}
		dictest = {}
		dicdata['ID'] = self.__ids
		dicdata['target'] = self.__targets
		
		dictest['ID'] =  self.__idstest

		for (feature_name, feature_samples) in self.__datatest.iteritems():
			dictest[feature_name] = feature_samples

		for (feature_name, feature_samples) in self.__data.iteritems():
			dicdata[feature_name] = feature_samples

	
		pd.DataFrame(dicdata).to_csv('../Input/without_nulls_train.csv',index=False)
		pd.DataFrame(dictest).to_csv('../Input/without_nulls_test.csv',index=False)

		return self.__data


	def get_lda(self,data):
		"""
		this function returns the array variables, for to use in the classifier
		"""
		print(data)

	def __DataFrame(self,data):
		'''
		this function returns the DataFrame, for to use into the classifier
		'''
		return pd.DataFrame(data)

	def __xgbclassifier(self):
		print("XGBOOST")
		'''
		classifier
		'''
		params = {
        'objective':'binary:logistic',
        'eval_metric':'logloss',
        'eta' :             0.07,
        'min_child_weight': 1,
        'subsample':        0.9,
        'colsample_bytree': 0.9,
        'max_depth':        10,
    	}

		cross_validation = 0.2
		n_cv    = int(self.n_samples * cross_validation)
		n_train = self.n_samples - n_cv

		cv    = self.__data[n_train:self.n_samples].copy()
		new_train = self.__data[:n_train].copy()
		target_cv = self.__data[n_train:self.n_samples].copy()
		target_train = self.__targets[:n_train].copy()
		print ('cv: ', cv.shape, '  --  cv-target: ', target_cv.shape)
		print ('train: ', new_train.shape, '  --  train-target: ', target_train.shape)

		xgtrain = xgb.DMatrix(new_train, target_train)
		print(cv)
		xgcv = xgb.DMatrix( cv, target_cv ) 
		print('sale 1')
		xgtest = xgb.DMatrix(self.__datatest)
		print('sale 2')
		num_round = 500
		plst = list(params.items())
		watchlist = [(xgtrain, 'train'),(xgcv, 'val')]
		bst = xgb.train( plst, xgtrain, num_round, watchlist, early_stopping_rounds=50)
		y_pred = bst.predict(xgtest)
		print('Testing... ')

		pd.DataFrame({"ID": id_test, "PredictedProb": y_pred}).to_csv('xgboost_test2.csv',index=False)

	def __svmclassifier(self):
		print('SVM')

	def classify(self,cla='xgb'):
		print('Classifying', end=' ')
		cl = {
			'smv' : self.__svmclassifier,
			'xgb' : self.__xgbclassifier
		}
		if cla in cl:
			cl[cla]()

