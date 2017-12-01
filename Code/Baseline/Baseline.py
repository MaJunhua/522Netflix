from surprise import SVDpp
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise import Reader
from surprise import BaselineOnly

import os 
path = "/Users/zhangqi/Google Drive/Study/CSC 522/522 Project/data/"
# os.chdir(path)

#define the data format to feed to the algorithm
reader = Reader(line_format='item user rating timestamp', sep=',')

def batchRunBaseline(data, al, folds):
	'''
    define a function to run batches of data
    Args:
        data: data file name in string.
        al: algorithm name in string.
        folds: split the data into x folds for cross-validation, interger
    Returns:
        None	
	'''

	#load the data with given data format
	print "load data..."
	data = Dataset.load_from_file(path + data, reader=reader)

	#split the data into x folds for cross-validation.
	print "Split data...."
	data.split(n_folds=folds)

	bsl_options = {'method': 'sgd',
               'learning_rate': .00005,
               }

	elif al == 'SVDpp':
		algo = SVDpp()
	elif al == 'Base':
		algo = BaselineOnly(bsl_options=bsl_options)

	# Evaluate performances of the algorithm on the dataset.
	perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

	print_perf(perf)


batchRunBaseline("/movieID_lessthan500.data",'Base',10)



