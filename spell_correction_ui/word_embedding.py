from pickle import load
from numpy import array
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from matplotlib import pyplot as plt
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
# from Capsule_Keras import *
from fasttext import FastText
# import fasttext
# import visualkeras
import keras
import numpy as np
from sklearn.manifold import TSNE
import random
import os

# Every data file and model artifact lives in spell_correction_ui/datasets/,
# which is gitignored - nothing under it is committed. Paths resolve relative to
# this file, so the script runs from any working directory. See the README for
# how to produce each input.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')

# Input: the cleaned corpus from neural_network.py.
BOTH_PKL = os.path.join(DATA_DIR, 'english-german-both.pkl')

# Input: pretrained Persian fastText vectors, 300-d.
# Download cc.fa.300.bin from https://fasttext.cc/docs/en/crawl-vectors.html
FASTTEXT_MODEL = os.path.join(DATA_DIR, 'cc.fa.300.bin')

# Optional, used only by the unused pre_train_glove() helper below.
GLOVE_FILE = os.path.join(DATA_DIR, 'glove.6B.100d.txt')

# Output: the embedding matrix consumed by network_architecture.py.
EMBEDDING_MATRIX = os.path.join(DATA_DIR, 'embedding_matrix.txt')

for _path, _made_by in ((BOTH_PKL, 'neural_network.py'),
                        (FASTTEXT_MODEL, 'downloading cc.fa.300.bin from fasttext.cc')):
    if not os.path.exists(_path):
        raise SystemExit(
            'Missing required file: ' + _path +
            ' -- produce it with ' + _made_by + ' (see the README).'
        )


# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


def pre_train_glove(vocab_size, word_index):
    embeddings_index = dict()
    f = open(GLOVE_FILE, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def pre_train_fasttext(words):
    fastText_model_path = FASTTEXT_MODEL
    model = FastText.load_model(fastText_model_path)
    embedding_matrix = zeros((len(words) + 1, 300))
    for i in range(0, len(words) - 1):
        embedding_matrix[i] = model.get_word_vector(words[i])
    return embedding_matrix


# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units, embedding_matrix):
    model = Sequential()
    # model.add(Embedding(src_vocab, n_units, input_length=src_timesteps))
    model.add(Embedding(src_vocab, 300, weights=[embedding_matrix], input_length=src_timesteps, trainable=False))
    # model.add(Conv1D(4 * n_units, 3, activation='relu', padding='same'))
    # model.add(MaxPooling1D(pool_size=2, padding='same'))
    # model.add(Capsule(num_capsule=10, dim_capsule=16, routings=3, share_weights = True))
    model.add(Bidirectional(LSTM(n_units)))
    model.add(RepeatVector(tar_timesteps))
    model.add(Bidirectional(LSTM(n_units, return_sequences=True)))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    # visualkeras.layered_view(model, to_file='output.png')
    return model


# load datasets
dataset = load_clean_sentences(BOTH_PKL)
# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
# print(eng_tokenizer.word_index.keys())
# print(eng_tokenizer.word_index)
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
# print('English Vocabulary Size: %d' % eng_vocab_size)
# print('English Max Length: %d' % (eng_length))
# prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
# print(ger_tokenizer.word_index)
wordFastTet=[]
for key in ger_tokenizer.word_index:
    wordFastTet.append(key)
# print(ger_tokenizer.word_index)
print('German Vocabulary Size: %d' % ger_vocab_size)
print('German Max Length: %d' % ger_length)
embedding_matrix = pre_train_fasttext(wordFastTet)
np.savetxt(EMBEDDING_MATRIX, embedding_matrix)
