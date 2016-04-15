import pandas as pd
import operator
import numpy as np
import cPickle
import scipy.sparse as sps
import itertools




def combining_features(file_input, vars = ['v50','v38','v129','v14','v85','v114','v75','v74','v125','v47','v71','v3','v18','v79','v30','v56','v24','v110','v91','v10'])
	data = pd.read_csv("../../../Input/train.csv");

	result = dict();
	column_names = list(corr.columns.values);
	for i in range(len(vars)) :
		column1 = df[i]
		for j in range(len(vars)) :
			column2 = df[j];
			new_column_name = column_names[i] + '_' + column_names[j];
			new_var = 
		new = reduce_lavels(fileInput, categorical[i], dictionary[i], threshold_values[i])
		result[categorical[i]] = new


if __name__ == '__main__':