import pandas as pd
import operator
import numpy as np
import scipy.sparse as sps
import itertools




def combining_features(file_input_train, file_input_test, file_output_train, file_output_test, vars = ['v50', 'v38', 'v34', 'v99', 'v129', 'v23', 'v21', 'v19', 'v72', 'v18', 'v12', 'v126', 'v102', 'v78', 'v114']) :
	train = pd.read_csv(file_input_train);
	id_train = train['ID'];
	target_train = train['target'];
	train = train.drop(['ID','target'],axis=1)

	test = pd.read_csv(file_input_test);
	id_test = test['ID'];
	test = test.drop(['ID'],axis=1)


	result_train = dict();
	result_test = dict();
	result_train['ID'] = id_train;
	result_train['target'] = target_train;
	result_test['ID'] = id_test;
	print ('len(vars): ', len(vars))
	for i in range(len(vars)) :
		column_train1 = train[vars[i]]
		column_test1 = test[vars[i]]
		print( 'column_train1.shape:', column_train1.shape)
		for j in range(len(vars)) :
			column_train2 = train[vars[j]];
			column_test2 = test[vars[j]]
			print ('vars[i]: ', vars[i], ', vars[j]: ', vars[j])
			new_column_header = vars[i] + '_' + vars[j];
			result_train[new_column_header] = column_train2.multiply(column_train1, 'columns');
			result_test[new_column_header] = column_test2.multiply(column_test1, 'columns');
			out_mean = result_train[new_column_header].mean();
			out_max = result_train[new_column_header].max(); 
			out_min = result_train[new_column_header].min();
			result_train[new_column_header] = (result_train[new_column_header] - out_mean) / (out_max - out_min) 
			result_test[new_column_header] = (result_test[new_column_header] - out_mean) / (out_max - out_min) 
			print( 'len(result_train): ', len(result_train[new_column_header]) );


	for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
		if train_series.dtype == 'O':
			result_train[train_name] = train[train_name]
			result_test[test_name] = test[test_name];
		else:
			out_mean = train[train_name].mean();
			out_max = train[train_name].max(); 
			out_min = train[train_name].min();
			result_train[train_name] = (train[train_name] - out_mean) / (out_max - out_min) 
			result_test[test_name] = (test[test_name] - out_mean) / (out_max - out_min) 
	
	output_train = pd.DataFrame(result_train);
	output_test = pd.DataFrame(result_test);

	print ( 'output_train.shape', output_train.shape );
	output_train.to_csv(file_output_train , index=False)

	del output_train
	del result_train
	del id_train
	del target_train
	del column_train1
	del column_train2

	print ( 'output_test.shape', output_test.shape );
	output_test.to_csv(file_output_test , index=False)

if __name__ == '__main__':
	print ('hello world')
	output_file_train = '../../../Input/combined_features_train2.csv';
	output_file_test = '../../../Input/combined_features_test2.csv';
	input_file_train = '../../../Input/jorge_train_001.csv';
	input_file_test = '../../../Input/jorge_test_001.csv';
	vars = ['v50', 'v38', 'v34', 'v99', 'v129', 'v23', 'v21', 'v19', 'v72', 'v18', 'v12', 'v126', 'v102', 'v78', 'v114', 'v10', 'v115', 'v39', 'v62', 'v124'];
	combining_features(input_file_train, input_file_test, output_file_train, output_file_test, vars =vars );