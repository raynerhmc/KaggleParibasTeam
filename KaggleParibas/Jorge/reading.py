import pandas as pd
import operator
import numpy as np



def read_csv(filename):
	'''
	Lee el csv y devuelve un diccionario con los valores medios o moda, max, min de las variables numericas y categoricas 
	'''
	df = pd.read_csv(filename)
	categorical = ['v3','v22','v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107' , 'v110', 'v112', 'v113', 'v125']
	result_values = dict()
	for i in categorical:
		dicc = dict()
		column = df[i]
		for j in column:
			if not dicc.has_key(j) and not pd.isnull(j):
				dicc[j] = 1
			elif dicc.has_key(j) and not pd.isnull(j):
				value = dicc.get(j)+1
				dicc[j] = value
		value = (max(dicc.iteritems(), key=operator.itemgetter(1))[0] , min(dicc.iteritems(), key=operator.itemgetter(1))[0]) 
		result_values[i] = value


	remove = categorical
	remove.extend(['ID','target'])
	numerical = df.drop(remove,axis=1)

	for i in numerical:
		column = df[i]
		vector = []
		for j in column:
			if not pd.isnull(j):
				vector.append(j)
		value = (np.mean(vector) , max(vector), min(vector))
		result_values[i] = value		

	return result_values


def complete_data(filenameInput, filenameOutput, categorical="max" , numerical="media"):
	'''
	Recibe el archivo de entrada y el nombre del nuevo archivo
	El parametro categorical por defecto "max" llena los nulos con la moda. "min" los llena con el valor minimo
	El parametro numerical por defecto "media" los llena con el promedio. 

	categorical: max - min 
	numerical: media , max , min 
	'''

	categorical_values = ['v3','v22','v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107' , 'v110', 'v112', 'v113', 'v125']


	result_values = read_csv(filenameInput)
	new_data = dict()	

	
	df = pd.read_csv(filenameInput)
	columns = [x for x in df] 

	index_categorical = 0
	index_numerical = 0

	if categorical=="max":
		index_categorical = 0
	else:
		index_categorical = 1


	if numerical == "media":
		index_numerical = 0
	elif numerical == "max":
		index_numerical = 1
	else: 
		index_numerical = 2


	for i in df:
		vector = []
		column =  df[i]

		if result_values.has_key(i):
			
			for j in column:
				if pd.isnull(j):
					value=0
					if i in categorical_values:
						value = result_values.get(i)[index_categorical]
					else:
						value = result_values.get(i)[index_numerical]
					vector.append(value)
				else:
					vector.append(j)
			new_data[i] = vector
		else:
			vector = [x for x in column]
			new_data[i] = vector
	
	data = pd.DataFrame(new_data, columns=columns)
	data.to_csv(filenameOutput) 
	return data

def complete_train_and_test(trainInput, trainOutput, testInput ,testOutput):
	categorical_values = ['v3','v22','v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107' , 'v110', 'v112', 'v113', 'v125']


	result_values = read_csv(trainInput)

	index_categorical = 0 # moda
	index_numerical = 0 # media


	new_data = dict()	
	new_data_test = dict()
	df = pd.read_csv(trainInput)
	columns = [x for x in df] 
	
	for i in df:
		vector = []
		column =  df[i]

		if result_values.has_key(i):
			
			for j in column:
				if pd.isnull(j):
					value=0
					if i in categorical_values:
						value = result_values.get(i)[index_categorical]
					else:
						value = result_values.get(i)[index_numerical]
					vector.append(value)
				else:
					vector.append(j)
			new_data[i] = vector
		else:
			vector = [x for x in column]
			new_data[i] = vector



	df2 = pd.read_csv(testInput)
	for i in df2:
		vector = []
		column = df2[i]

		if result_values.has_key(i):
			for j in column:
				if pd.isnull(j):
					value = 0
					if i in categorical_values:
						value = result_values.get(i)[index_categorical]
					else:
						value = result_values.get(i)[index_numerical]
					vector.append(value)
				else:
					vector.append(j)
			new_data_test[i] = vector 
		else:
			vector = [x for x in column]
			new_data_test[i] = vector
	




	data = pd.DataFrame(new_data, columns=columns)
	data.to_csv(trainOutput)


	columns.remove('target')
	data2 = pd.DataFrame(new_data_test, columns=columns)	
	data2.to_csv(testOutput)

 

if __name__ == '__main__':

	data_train = "data/train.csv"
	data_test = "data/test.csv"


	muestra = "data/muestra.csv"
	muestra_test = "data/muestra_test.csv"
	
	#data = complete_data(muestra , "output3.csv", categorical="min", numerical="min")
	#data = complete_data(data_train , "outputTrain.csv")


	complete_train_and_test(muestra, "outputMuestraTrain.csv", muestra_test, "outputMuestraTest.csv")
