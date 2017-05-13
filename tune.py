import pandas as pd
import numpy as np
import random
from model import CNNModel, Preprocessor
import cPickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def read_word_embeddings():
	wve = {}
	fread1 = open('../data/vectors.txt', 'r')
	fread2 = open('../data/types.txt', 'r')
	
	while True:
		word = fread2.readline().strip()
		# print word
		# print fread1.readline().strip().split(" ")
		
		try:
			embedding = [float(l) for l in fread1.readline().strip().split(" ")]
			embedding = np.array(embedding)/np.linalg.norm(embedding)
			embedding = embedding.tolist()
			wve[word] = embedding
		except ValueError:
			print "skipped"

		if not word:
			break
	return wve


def training(model, traindata, Y):
	titles, abstracts, topics = traindata
	topic_input = model.preprocessor_topic.build_sequences(topics)
	title_input = model.preprocessor_title.build_sequences(titles)
	abstract_input = model.preprocessor_abstract.build_sequences(abstracts)

	size = int(0.8*len(Y)) 
	ind = random.sample(xrange(len(Y)), size)
	indc = [i for i in xrange(len(Y)) if i not in ind]

	topic_input_train = topic_input[ind, :]
	title_input_train = title_input[ind, :]
	abstract_input_train = abstract_input[ind, :]
	Y_train = Y[ind]

	topic_input_val = topic_input[indc, :]
	title_input_val = title_input[indc, :]
	abstract_input_val = abstract_input[indc, :]
	Y_val = Y[indc]

	print "train true label fraction:", np.sum(Y)/float(len(Y))
	model.train(topic_input_train, title_input_train, abstract_input_train, Y_train, topic_input_val, title_input_val, abstract_input_val, Y_val, nb_epoch=50, batch_size=200)
	return model


def testing(model, testdata):
	titles, abstracts, topics = testdata
	topic_input = model.preprocessor_topic.build_sequences(topics)
	title_input = model.preprocessor_title.build_sequences(titles)
	abstract_input = model.preprocessor_abstract.build_sequences(abstracts)

	Y = model.predict(topic_input, title_input, abstract_input, batch_size=200)

	return Y


def drawplot(Y_prediction, Y_true):
	Y_prediction = np.array(Y_prediction).flatten()
	Y_true = np.array(Y_true).flatten()
	pred = Y_true[np.argsort(Y_prediction)[::-1]]
	pred = np.cumsum(pred)
	print pred
	plt.plot(pred)
	plt.savefig('graph.png')

if __name__ == '__main__':
	print "here"
	df1 = pd.read_csv('dta_data.csv')
	df2 = pd.read_csv('review_to_title.csv')
	df3 = pd.merge(df1, df2, on='review_id', how='inner')
	print "starting replacing"
	df3 = df3.replace(np.nan, " ", regex=True)
	print "done with replacing"
	titles = df3['title'].values.tolist()
	abstracts = df3['abstract'].values.tolist()
	topics = df3['review_topic'].values.tolist()
	Y = df3['fulltext_relevant'].values
	# Y = df3['abstract_relevant'].values
	print "reading embeddings"
	wve = read_word_embeddings()
	print "done reading embeddings"
	preprocessor_title = Preprocessor(max_features=14000, maxlen=20, wvs=wve)
	preprocessor_abstract = Preprocessor(max_features=14000, maxlen=200, wvs=wve)
	preprocessor_topic = Preprocessor(max_features=166, maxlen=20, wvs=wve)
	
	preprocessor_topic.preprocess(topics)
	preprocessor_title.preprocess(titles)
	preprocessor_abstract.preprocess(abstracts)

	print "creatign model"
	model = CNNModel(preprocessor_title, preprocessor_abstract, preprocessor_topic, dropout=0.7)
	print "done creatign model"

	trainsize = int(0.6*len(Y))
	indtrain = random.sample(xrange(len(Y)),trainsize)
	indtest = np.array([i for i in xrange(len(Y)) if i not in indtrain])
	indtrain = np.array(indtrain)
	titles_train = []
	abstracts_train = []
	topics_train = []
	Y_train = []
	
	for i in indtrain:
		titles_train.append(titles[i])
		topics_train.append(topics[i])
		abstracts_train.append(abstracts[i])
		Y_train.append(Y[i])
	titles_test = []
	abstracts_test = []
	topics_test = []
	Y_true = []
	for i in indtest:
		titles_test.append(titles[i])
		topics_test.append(topics[i])
		abstracts_test.append(abstracts[i])
		Y_true.append(Y[i])

	Y_train = np.array(Y_train)
	Y_true = np.array(Y_true)

	print "starting model training"
	model = training(model, [titles_train, abstracts_train, topics_train], Y_train)
	print "done model training"

	print "starting testing"
	Y_prediction = testing(model, [titles_test, abstracts_test, topics_test])
	print "done testing"

	# print "writing test results to file"
	cPickle.dump((Y_prediction, Y_true), open('results.pkl', 'w'))
	drawplot(Y_prediction, Y_true)
	# print "done with test results to file"



