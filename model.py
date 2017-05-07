import numpy as np
from keras.preprocessing import sequence
from keras.layers import Merge, Input, Dense, merge
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.models import Model

class CNNModel:

    def __init__(self, preprocessor_title, preprocessor_abstract, preprocessor_topic, filters=None, n_filters=100, dropout=0.0):
        '''
        parameters
        ---
        preprocessor: an instance of the Preprocessor class, defined below
        '''
        self.preprocessor_topic = preprocessor_topic
        self.preprocessor_title = preprocessor_title
        self.preprocessor_abstract = preprocessor_abstract

        if filters is None:
            self.ngram_filters = [3, 4, 5]
        else:
            self.ngram_filters = filters 

        self.nb_filter = n_filters 
        self.dropout = dropout

        self.build_model() # build model
    
    def train(self, topic_train, title_train, abstract_train, y_train, topic_val=None, title_val=None, abstract_val=None, y_val=None,
                nb_epoch=5, batch_size=32, optimizer='adam'):

        checkpointer = ModelCheckpoint(filepath="weights.hdf5", 
                                       verbose=1, 
                                       save_best_only=(topic_val is not None))

        if topic_val is not None:
            self.model.fit([topic_train, title_train, abstract_train], [y_train],
                batch_size=batch_size, nb_epoch=nb_epoch,
                validation_data=([topic_val, title_val, abstract_val],  [y_val]),
                verbose=2, callbacks=[checkpointer])
        else: 
            print("no validation data provided!")
            self.model.fit([topic_train, title_train, abstract_train], [y_train],
                batch_size=batch_size, nb_epoch=nb_epoch, 
                verbose=2, callbacks=[checkpointer])

    def predict(self, title, topic, abstract, batch_size=32, binarize=False):
        raw_preds = self.model.predict([title, topic, abstract], batch_size=batch_size)

        if binarize:
          return np.round(raw_preds)
        return raw_preds


    def build_model(self):
        topic_input = Input(shape=(self.preprocessor_topic.maxlen,), dtype='int32')
        x = Embedding(output_dim=200, input_dim=self.preprocessor_topic.max_features, 
            input_length=self.preprocessor_topic.maxlen, weights=self.preprocessor_topic.init_vectors)(topic_input)
        y1 = Convolution1D(nb_filter=self.nb_filter,
                                         filter_length=3,
                                         border_mode='valid',
                                         activation='relu',
                                         subsample_length=1,
                                         input_dim=200,
                                         input_length=self.preprocessor_topic.maxlen)(x)
        y1 = MaxPooling1D(pool_length=self.preprocessor_topic.maxlen - 3 + 1)(y1)
        y1 = Flatten()(y1)

        title_input = Input(shape=(self.preprocessor_title.maxlen,), dtype='int32')
        x = Embedding(output_dim=200, input_dim=self.preprocessor_title.max_features, 
            input_length=self.preprocessor_title.maxlen, weights=self.preprocessor_title.init_vectors)(title_input)
        y2 = Convolution1D(nb_filter=self.nb_filter,
                                         filter_length=3,
                                         border_mode='valid',
                                         activation='relu',
                                         subsample_length=1,
                                         input_dim=200,
                                         input_length=self.preprocessor_title.maxlen)(x)
        y2 = MaxPooling1D(pool_length=self.preprocessor_title.maxlen - 3 + 1)(y2)
        y2 = Flatten()(y2)

        abstract_input = Input(shape=(self.preprocessor_abstract.maxlen,), dtype='int32')
        x = Embedding(output_dim=200, input_dim=self.preprocessor_abstract.max_features, 
            input_length=self.preprocessor_abstract.maxlen, weights=self.preprocessor_abstract.init_vectors)(abstract_input)
        y3 = Convolution1D(nb_filter=self.nb_filter,
                                         filter_length=3,
                                         border_mode='valid',
                                         activation='relu',
                                         subsample_length=1,
                                         input_dim=200,
                                         input_length=self.preprocessor_abstract.maxlen)(x)
        y3 = MaxPooling1D(pool_length=self.preprocessor_abstract.maxlen - 3 + 1)(y3)
        y3 = Flatten()(y3)
        z = merge([y1, y2, y3], mode='concat')
        z = Dropout(self.dropout)(z)
        z = Dense(200, input_dim=self.nb_filter * len(self.ngram_filters))(z)
        final_output = Dense(1, activation='sigmoid')(z)
        self.model = Model(input=[topic_input, title_input, abstract_input], output=[final_output])
        self.model.compile(loss='binary_crossentropy', optimizer="rmsprop")
        print("model built")
        print(self.model.summary())

class Preprocessor:
    def __init__(self, max_features, maxlen, embedding_dims=200, wvs=None):
        self.max_features = max_features  
        self.tokenizer = Tokenizer(nb_words=self.max_features)
        self.maxlen = maxlen  

        self.use_pretrained_embeddings = False 
        self.init_vectors = None 
        if wvs is None:
            self.embedding_dims = embedding_dims
        else:
            # note that these are only for initialization;
            # they will be tuned!
            self.use_pretrained_embeddings = True
            self.embedding_dims = len(wvs[wvs.keys()[0]])
            self.word_embeddings = wvs


    def preprocess(self, all_texts):
       
        self.raw_texts = all_texts
        #self.build_sequences()
        self.fit_tokenizer()
        if self.use_pretrained_embeddings:
            self.init_word_vectors()

    def fit_tokenizer(self):
        self.tokenizer.fit_on_texts(self.raw_texts)
        self.word_indices_to_words = {}
        for token, idx in self.tokenizer.word_index.items():
            self.word_indices_to_words[idx] = token

    def build_sequences(self, texts):
        X = list(self.tokenizer.texts_to_sequences_generator(texts))
        X = np.array(pad_sequences(X, maxlen=self.maxlen))
        return X

    def init_word_vectors(self):
        self.init_vectors = []
        unknown_words_to_vecs = {}
        for t, token_idx in self.tokenizer.word_index.items():
            if token_idx <= self.max_features:
                try:
                    self.init_vectors.append(self.word_embeddings[t])
                except:
                    if t not in unknown_words_to_vecs:
                        # randomly initialize
                        unknown_words_to_vecs[t] = np.random.random(
                                                self.embedding_dims)*-2 + 1

                    self.init_vectors.append(unknown_words_to_vecs[t])

        self.init_vectors = [np.vstack(self.init_vectors)]
