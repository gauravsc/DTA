import pandas as pd
import numpy as np

files = ['results_abstract_relevant_test.csv', 'results_fulltext_relevant_test.csv']
new_files = ['run_abstract_test.csv', 'run_fulltext_test.csv']

for i in range(len(files)):
	file = files[i]
	df = pd.read_csv(file)
	df1 = df[['review_id', 'pmid', 'yprediction']]
	df1['RANK'] = np.argsort(np.argsort(-df1['yprediction']))
	df1['INTERACTION'] = ""
	df1['RUN-ID'] = 0
	df1 = df1.rename(columns = {'review_id':'TOPIC-ID', 'pmid':'PID', 'yprediction':'SCORE', 
		'RANK':'RANK', 
		'INTERACTION':'INTERACTION',
		'RUN-ID':'RUN-ID'
		})
	df1[['TOPIC-ID', 'PID', 'RANK', 'SCORE', 'RUN-ID']].to_csv(new_files[i], index=False, sep=' ', header=False)