import pandas as pd
import operator
import numpy as np
import cPickle
import scipy.sparse as sps
import itertools


data_train = "data/train.csv"
data_test = "data/test.csv"
muestra = "data/muestra.csv"
fileDict = "diccionario.pk"

'''
write data y load son funciones para guardar y cargar cualquier informacion en disco
'''

def write_data_to_disk(file, data):
    with open(file, 'wb') as fid:
        cPickle.dump(data, fid)

def load_data_from_disk(file):
    with open(file, 'rb') as fid:
        data = cPickle.load(fid)
    return data

'''
Para saber el porcentaje de frecuencia de cada uno de los niveles de cada variable
La funcion devuelve un vector de diccionarios (cada diccionario por variable)
Alternativamente, el vector es guardado en disco. 
'''
def statistics_v2(fileInput, fileOutput):
	categorical = ['v3','v22','v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107' , 'v110', 'v112', 'v113', 'v125']
	size_data = 114321

	df = pd.read_csv(fileInput)
	vector_dicts = []

	for i in categorical:
		values = dict()
		column = df[i]
		for j in column:
			if values.has_key(j):
				values[j] = values.get(j) + 1
			else:
				values[j] = 1

		new_dicc = dict()
		for  j  in values.items():
			percentage = j[1] * 100 / float(size_data) 
			if not pd.isnull(j[0]):
				new_dicc[j[0]] = percentage
			else:
				new_dicc["null"] = percentage

		vector_dicts.append(new_dicc)

	write_data_to_disk(fileOutput, vector_dicts) 
			

'''
Recibe una categoria y reduce los niveles que tenga de acuerdo al porcentaje de frecuencia que es enviado como parametro
Valores con porcentaje de frecuencia menor al parametro threshold son combinados en un nuevo level.
'''
def reduce_lavels(fileInput, category ,dictionary, threshold=0):

	#moda = max(dictionary.iteritems(), key=operator.itemgetter(1))[0]
	#print moda
	df = pd.read_csv(fileInput)
	column = df[category]
	new_values = []
	
	for i in column:
		if pd.isnull(i):
			i = "null"		
		per_value = dictionary.get(i)
		if per_value > threshold:
			new_values.append(i)						
		else:
			new_values.append("New_var")			

	return new_values 

'''
Reduce los niveles para cada categoria y guarda la nueva tabla en  un CSV
Se mando un conjunto de threshold values de acuerdo al analisis de cada variable. Aun no estoy seguro de cual seria el valor ideal a mandar
'''
def manage_data(fileInput, fileOutput):
	categorical = ['v3','v22','v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107' , 'v110', 'v112', 'v113', 'v125']
	threshold_values = [3, 0.085, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 2, 2, 0, 1.5, 0.5, 0.16]
	dictionary = load_data_from_disk(fileDict)

	result = dict()
	
	for i in range(len(categorical)):
		new = reduce_lavels(fileInput, categorical[i], dictionary[i], threshold_values[i])
		result[categorical[i]] = new		 


	df = pd.read_csv(fileInput)
	remove = categorical
	numerical = df.drop(remove,axis=1)
	for i in numerical:
		vector = []
		column = df[i]
		for j in column:
			vector.append(j)
		result[i] = vector


	columns = [x for x in df] 
	data = pd.DataFrame(result, columns=columns)


	oneHot = pd.get_dummies(data)
	oneHot.to_csv(fileOutput)
	return oneHot


def manage_train_test(inputTrain, outputTrain , inputTest, outputTest):
	categorical = ['v3','v22','v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107' , 'v110', 'v112', 'v113', 'v125']
	threshold_values = [3, 0.085, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 2, 2, 0, 1.5, 0.5, 0.16]
	dictionary = load_data_from_disk(fileDict)

	
	result = dict()	
	for i in range(len(categorical)):
		new = reduce_lavels(inputTrain, categorical[i], dictionary[i], threshold_values[i])
		result[categorical[i]] = new		 

	df = pd.read_csv(inputTrain)
	remove = categorical
	numerical = df.drop(remove,axis=1)
	for i in numerical:
		vector = []
		column = df[i]
		for j in column:
			vector.append(j)
		result[i] = vector


	columns = [x for x in df] 
	data = pd.DataFrame(result, columns=columns)
	oneHot = pd.get_dummies(data)
	oneHot.to_csv(outputTrain)
	


	

	result = dict()	
	for i in range(len(categorical)):
		new = reduce_lavels(inputTest, categorical[i], dictionary[i], threshold_values[i])
		result[categorical[i]] = new		 

	df = pd.read_csv(inputTest)
	remove = categorical
	numerical = df.drop(remove,axis=1)

	for i in numerical:
		vector = []
		column = df[i]
		for j in column:
			vector.append(j)
		result[i] = vector

	columns = [x for x in df] 
	

	
	data = pd.DataFrame(result, columns=columns)
	oneHot = pd.get_dummies(data)
	oneHot.to_csv(outputTest)
	
	

	







	#return oneHot


	

if __name__ == '__main__':
	
	#statistics(muestra)
	#statistics_v2("outputTrain.csv" , fileDict)
	
	#dicc = load_data_from_disk(fileDict)
	#print dicc[0]


	#manage_data(data_train, "probaFinal.csv")

	manage_train_test("outputTrain.csv", "readyTrain.csv" , "outputTest.csv", "readyTest.csv")



