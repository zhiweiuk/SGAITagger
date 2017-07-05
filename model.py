from __future__ import print_function, unicode_literals
import numpy as np
np.random.seed(1337)  # for reproducibility

import os

from six.moves import zip
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop, SGD
from keras.utils.data_utils import get_file
from keras.callbacks import Callback
from subprocess import Popen, PIPE, STDOUT
from colors import green, red, blue

import utils.glove_conll2003 as conll2003
from utils.ChainCRF import *

def run_conlleval(X_words_test, y_test, y_pred, index2word, index2chunk, pad_id=0):
    '''
    Runs the conlleval script for evaluation the predicted IOB-tags.
    '''
    url = 'http://www.cnts.ua.ac.be/conll2000/chunking/conlleval.txt'
    path = get_file('conlleval',
                    origin=url,
                    md5_hash='61b632189e5a05d5bd26a2e1ec0f4f9e')

    p = Popen(['perl', path], stdout=PIPE, stdin=PIPE, stderr=STDOUT)

    y_true = np.squeeze(y_test, axis=2)

    sequence_lengths = np.argmax(X_words_test == pad_id, axis=1)
    nb_samples = X_words_test.shape[0]
    conlleval_input = []
    for k in range(nb_samples):
        sent_len = sequence_lengths[k]
        words = list(map(lambda idx: index2word[idx], X_words_test[k][:sent_len]))
        true_tags = list(map(lambda idx: index2chunk[idx], y_true[k][:sent_len]))
        pred_tags = list(map(lambda idx: index2chunk[idx], y_pred[k][:sent_len]))
        sent = zip(words, true_tags, pred_tags)
        for row in sent:
            conlleval_input.append(' '.join(row))
        conlleval_input.append('')
    print()
    conlleval_stdout = p.communicate(input='\n'.join(conlleval_input).encode())[0]
    print(blue(conlleval_stdout.decode()))
    print()


class ConllevalCallback(Callback):
    '''Callback for running the conlleval script on the test dataset after
    each epoch.
    '''
    def __init__(self, X_test, y_test, batch_size=1, index2word=None, index2ner=None):
        self.X_words_test, self.X_pos_test, self.X_chunk_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.index2word = index2word
        self.index2ner = index2ner

    def on_epoch_end(self, epoch, logs={}):
        X_test = [self.X_words_test, self.X_pos_test, self.X_chunk_test]
        pred_proba = model.predict(X_test)
        y_pred = np.argmax(pred_proba, axis=2)
        run_conlleval(self.X_words_test, self.y_test, y_pred, self.index2word, self.index2ner)


maxlen = 80  # cut texts after this number of words (among top max_features most common words)
word_embedding_dim = 100
pos_embedding_dim = 8
chunk_embedding_dim = 6
lstm_dim = 100
batch_size = 64

print('Loading data...')
(X_words_train, X_pos_train, X_chunk_train, y_train), (X_words_dev, X_pos_dev, X_chunk_dev, y_dev), (X_words_test, X_pos_test, X_chunk_test, y_test), (index2word, index2pos, index2chunk, index2ner), (embedding_matrix) = conll2003.load_data(word_preprocess=lambda w: w.lower())

max_features = len(index2word)
nb_pos_tags = len(index2pos)
nb_chunk_tags = len(index2chunk)
nb_ner_tags = len(index2ner)

X_words_train = sequence.pad_sequences(X_words_train, maxlen=maxlen, padding='post')
X_pos_train = sequence.pad_sequences(X_pos_train, maxlen=maxlen, padding='post')
X_chunk_train = sequence.pad_sequences(X_chunk_train, maxlen=maxlen, padding='post')

X_words_dev = sequence.pad_sequences(X_words_dev, maxlen=maxlen, padding='post')
X_pos_dev = sequence.pad_sequences(X_pos_dev, maxlen=maxlen, padding='post')
X_chunk_dev = sequence.pad_sequences(X_chunk_dev, maxlen=maxlen, padding='post')

X_words_test = sequence.pad_sequences(X_words_test, maxlen=maxlen, padding='post')
X_pos_test = sequence.pad_sequences(X_pos_test, maxlen=maxlen, padding='post')
X_chunk_test = sequence.pad_sequences(X_chunk_test, maxlen=maxlen, padding='post')

y_train = sequence.pad_sequences(y_train, maxlen=maxlen, padding='post')
y_train = np.expand_dims(y_train, -1)

y_dev = sequence.pad_sequences(y_dev, maxlen=maxlen, padding='post')
y_dev = np.expand_dims(y_dev, -1)

y_test = sequence.pad_sequences(y_test, maxlen=maxlen, padding='post')
y_test = np.expand_dims(y_test, -1)

print('Unique words:', max_features)
print('Unique POS tags:', nb_pos_tags)
print('Unique Chunk tags:', nb_chunk_tags)
print('Unique NER tags:', nb_ner_tags)
print('X_words_train shape:', X_words_train.shape)
print('X_words_dev shape:', X_words_dev.shape)
print('X_words_test shape:', X_words_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print(">>" * 20)
print('Building model...')
print("<<" * 20)

# Word Input / Embedding Conversion
word_input = Input(shape=(maxlen,), dtype='int32', name='word_input')
word_emb = Embedding(
        max_features + 1,
        word_embedding_dim,
        weights=[embedding_matrix],
        trainable=False,
        input_length=maxlen,
        name='word_emb'
    )(word_input)

# POS Input / Embedding Conversion
pos_input = Input(shape=(maxlen,), dtype='int32', name='pos_input')
pos_emb = Embedding(nb_pos_tags, pos_embedding_dim, input_length=maxlen, name='pos_emb')(pos_input)

# Chunk Input / Embedding Conversion
chunk_input = Input(shape=(maxlen,), dtype='int32', name='chunk_input')
chunk_emb = Embedding(nb_chunk_tags, chunk_embedding_dim, input_length=maxlen, name='chunk_emb')(chunk_input)

total_emb = concatenate([word_emb, pos_emb, chunk_emb])

bilstm = Bidirectional(LSTM(lstm_dim, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=True))(total_emb)
x = Dropout(0.4)(bilstm)
dense = TimeDistributed(Dense(nb_ner_tags))(x)

crf = ChainCRF()
crf_output = crf(dense)

model = Model(inputs=[word_input, pos_input, chunk_input], outputs=[crf_output])

model.compile(loss=crf.sparse_loss,
              optimizer=RMSprop(lr=0.005),
              metrics=['sparse_categorical_accuracy'])

model.summary()

conlleval = ConllevalCallback(
                [X_words_test, X_pos_test, X_chunk_test],
                y_test,
                index2word=index2word,
                index2ner=index2ner,
                batch_size=batch_size
            )

print(">>" * 20)
print('Training model...')
print("<<" * 20)

model.fit(
    [X_words_train, X_pos_train, X_chunk_train],
    y_train,
    validation_data=([X_words_dev, X_pos_dev, X_chunk_dev], y_dev),
    batch_size=batch_size,
    epochs=10,
    callbacks=[conlleval]
)

