import pandas as pd
import numpy as np
import random
from model import CNNModel, Preprocessor
import cPickle
#import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


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

if __name__ == '__main__':
	df1 = pd.read_csv('dta_data.csv')
	df2 = pd.read_csv('review_to_title.csv')
	df3 = pd.merge(df1, df2, on='review_id', how='inner')
	df3 = df3.replace(np.nan, " ", regex=True)
	titles_train = df3['title'].values.tolist()
	abstracts_train = df3['abstract'].values.tolist()
	topics_train = df3['review_topic'].values.tolist()
	# Y_train = df3['fulltext_relevant'].values
	Y_train = df3['abstract_relevant'].values
	print "reading embeddings"
	wve = read_word_embeddings()
	print "done reading embeddings"

	df4 = pd.read_csv('dta_data_test.csv')
	df5 = pd.read_csv('review_to_title.csv')
	df6 = pd.merge(df4, df5, on='review_id', how='inner')
	df6 = df6.replace(np.nan, " ", regex=True)

	titles_test = df6['title'].values.tolist()
	abstracts_test = df6['abstract'].values.tolist()
	topics_test = df6['review_topic'].values.tolist()


	preprocessor_title = Preprocessor(max_features=14000, maxlen=20, wvs=wve)
	preprocessor_abstract = Preprocessor(max_features=14000, maxlen=200, wvs=wve)
	preprocessor_topic = Preprocessor(max_features=166, maxlen=20, wvs=wve)
	
	preprocessor_topic.preprocess(topics_train+topics_test)
	preprocessor_title.preprocess(titles_train+titles_test)
	preprocessor_abstract.preprocess(abstracts_train+abstracts_test)

	print "creatign model"
	model = CNNModel(preprocessor_title, preprocessor_abstract, preprocessor_topic, dropout=0.6)
	print "done creatign model"

	print "starting model training"
	model = training(model, [titles_train, abstracts_train, topics_train], Y_train)
	print "done model training"

	print "starting testing"
	Ypred = testing(model, [titles_test, abstracts_test, topics_test])
	print "done testing"

	df6['yprediction'] = Ypred
	df6.to_csv('results_abstract_relevant_test.csv', sep=',')


