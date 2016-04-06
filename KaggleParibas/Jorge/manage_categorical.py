import pandas as pd
import operator
import numpy as np


data_train = "data/train.csv"
data_test = "data/test.csv"
muestra = "data/muestra.csv"



def statistics(fileInput):
	categorical = ['v3','v22','v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107' , 'v110', 'v112', 'v113', 'v125']
	#categorical = ['v3' , 'v24']
	#size_data = 199
	size_data = 114321

	df = pd.read_csv(fileInput)

	for i in categorical:
		values = dict()
		column = df[i]
		for j in column:
			if values.has_key(j):
				values[j] = values.get(j) + 1
			else:
				values[j] = 1

		sorted_dicc = sorted(values.items(), key=lambda x: (-x[1], x[0]))
		print i 
		for  j  in sorted_dicc:
			percentage = j[1] * 100 / float(size_data) 
			print j , percentage
		print " "
		







if __name__ == '__main__':
	
	statistics(data_train)