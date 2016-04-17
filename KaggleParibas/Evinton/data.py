import pandas as pd
import operator
import numpy as np
from scipy import stats
import csv
import math
from sklearn.decomposition import PCA, KernelPCA
from operator import itemgetter
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

		#drop it 
		self.__train = self.__train.drop( ['ID','target'],axis=1)

		self.n_samples = self.__train.shape[0]
		self.n_features = self.__train.shape[1]
		self.__fill = False
		print('Processing ',self.n_samples * self.n_features," data")

	def __fill_data_mean(self):
		'''
		filling the data with means value for the numerical variable and mode value for the categorical variable
		'''
		dic = {}
		#handle the moda value for the categorical name
		for (feature_name, feature_samples) in self.__train.iteritems():
			if feature_samples.dtype == 'object':
				self.__train.loc[feature_samples.isnull(),feature_name] = feature_samples.mode()[0]
				#self.__train.loc[feature_samples.isnull(),feature_name]= stats.mode(feature_samples)
			else:
				tmp_len = len(self.__train[feature_samples.isnull()])
				if tmp_len>0:
					self.__train.loc[feature_samples.isnull(), feature_name] = np.mean(feature_samples)

			dic[feature_name] = self.__train[feature_name]

		self.__data = self.__DataFrame(dic)
		return self.__data

	def __fill_data_meannull(self):
		'''
		filling the data with means value for the numerical variable and mode value for the categorical variable
		'''
		dic = {}
		#handle the moda value for the categorical name
		for (feature_name, feature_samples) in self.__train.iteritems():
			if feature_samples.dtype == 'object':
				self.__train.loc[feature_samples.isnull(),feature_name] = 'null'
				#self.__train.loc[feature_samples.isnull(),feature_name]= stats.mode(feature_samples)
			else:
				tmp_len = len(self.__train[feature_samples.isnull()])
				if tmp_len>0:
					self.__train.loc[feature_samples.isnull(), feature_name] = np.mean(feature_samples)

			dic[feature_name] = self.__train[feature_name]

		self.__data = self.__DataFrame(dic)
		return self.__data

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
		for (train_name, train_series) in self.__train.iteritems():
			if train_series.dtype == 'object':
				#for objects: factorize
				self.__train[train_name], tmp_indexer = pd.factorize(self.__train[train_name])
				dic[train_name] = self.__train[train_name]
        		#but now we have -1 values (NaN)
			else:
				#for int or float: fill NaN
				tmp_len = len(self.__train[train_series.isnull()])
				
				if tmp_len>0:
					#print "mean", train_series.mean()
					self.__train.loc[train_series.isnull(), train_name] = -999
				dic[train_name] = self.__train[train_name]
		
		self.__data = self.__DataFrame(dic)
		return self.__data

	def fill_data(self,tfill = 'data'):
		self.__fill = True
		print("Filling the data")
		opt = {
			'dummy':self.__fill_data_dummy,
			'mean': self.__fill_data_mean,
			'meannull': self.__fill_data_meannull,
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
		print(corr.shape[0],' ',corr.shape[1])
		for i in range( 0,corr.shape[0] ):
			for j in range( 0, corr.shape[1] ):
			# correlation over 90% and under 100%
				if i != j and corr.iloc[i][j] > threshold and corr.iloc[i][j] < limit:
					strvar = 'v'+str(i)
					if  strvar not in corrleted:
						corrleted[strvar] = 1
					else:
						corrleted[strvar] = corrleted[strvar] + 1

		
		corrleted = sorted(corrleted.items(),key=itemgetter(1), reverse=True)
		for (feature, count) in corrleted:
			print(feature,' = ', count)
					# print only one correlation example 2x3 and not 3x2 because it's the same.
					#print ('positive correlation between: v', str(i), ' and v' , str(j) , ' --> corr: ', corr.iloc[i][j] )
					 	
	def __numcorrelated(self):
		return True

	def __catcorrelated(self):
		return True

	def __dropvariable(self,data):
		self.__data = self.__data.drop(data,axis=1)


	def get_pca(self,data=None):
		"""
		this function returns the array variables, for to use in the classifier
		"""
		# fit the model to data 
		pca = PCA()
		self.__data = pca.fit_transform(self.__data)
		return __DataFrame(self.__data)
		#transform the data 
	def get_kpca(self,data):
		"""
		this function returns the array variables, for to use in the classifier
		"""
		kpca = KernelPCA()
		self.__data = kpca.fit_transform(self.__data)
		return __DataFrame(self.__data)


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