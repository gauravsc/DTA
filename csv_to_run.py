import pandas as pd
import numpy as np

files = ['results_abstract_relevant.csv', 'results_fulltext_relevant.csv']
new_files = ['run_abstract.csv', 'run_fulltext.csv']

for i in range(len(files)):
	file = files[i]
	df = pd.read_csv(file)
	df1 = df[['review_id', 'pmid', 'yprediction']]
	df1['RANK'] = np.argsort(df1['yprediction'])
	df1['INTERACTION'] = ""
	df1['RUN-ID'] = 0
	df1 = df1.rename(columns = {'review_id':'TOPIC-ID', 'pmid':'PID', 'yprediction':'SCORE', 
		'RANK':'RANK', 
		'INTERACTION':'INTERACTION',
		'RUN-ID':'RUN-ID'
		})
	df1.to_csv(new_files[i], index=False, sep=',')