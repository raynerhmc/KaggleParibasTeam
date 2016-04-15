import pandas as pd
import operator
import numpy as np
import cPickle
import scipy.sparse as sps
import itertools




def combining_features(file_input, vars = ['v50','v38','v129','v14','v85','v114','v75','v74','v125','v47','v71','v3','v18','v79','v30','v56','v24','v110','v91','v10']) :
	categorical = ['v3','v22','v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107' , 'v110', 'v112', 'v113', 'v125']
	data = pd.read_csv("../../../Input/train.csv");

	result = dict();
	column_names = list(data.columns.values);

	for i in range(len(vars)) :
		column1 = data[vars[i]]
		for j in range(len(vars)) :
			column2 = data[vars[j]];
			new_column_header = column_names[i] + '_' + column_names[j];
			result[new_column_header] = column2.multiply(column1);


	output = pd.DataFrame(result, columns=columns)


if __name__ == '__main__':
	print ('hello world')